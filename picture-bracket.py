import streamlit as st
import random
import json
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import io
import pandas as pd

MAX_ENTRIES = 64
HISTORY_FILE = Path("bracket_history.json")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------- Persistence ----------------

def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ---------------- State ----------------

def ensure_state():
    defaults = {
        "entries": [],                  # list of {id, title, image_kind, image_ref}
        "bracket_rounds": [],
        "current_round": 0,
        "current_match": 0,
        "next_round_winners": [],
        "champion_index": None,
        "bracket_started": False,
        "winner_saved": False,
        "history": load_history(),
        "vote_log": [],
        "current_match_context": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_bracket_only():
    st.session_state.bracket_rounds = []
    st.session_state.current_round = 0
    st.session_state.current_match = 0
    st.session_state.next_round_winners = []
    st.session_state.champion_index = None
    st.session_state.bracket_started = False
    st.session_state.winner_saved = False
    st.session_state.vote_log = []
    st.session_state.current_match_context = None


def reset_everything():
    st.session_state.entries = []
    reset_bracket_only()

# ---------------- Image display helpers ----------------

def image_preview(img, max_width=200):
    st.image(img, width=max_width)

def image_vote(img, max_width=350):
    st.image(img, width=max_width)

# ---------------- Bracket logic ----------------

def generate_bracket():
    entries = st.session_state.entries
    n = len(entries)
    if n < 2:
        return

    indices = list(range(n))
    random.shuffle(indices)

    # pad to power of 2 with None
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    while len(indices) < next_pow2:
        indices.append(None)

    first_round = []
    for i in range(0, len(indices), 2):
        first_round.append((indices[i], indices[i + 1]))

    st.session_state.bracket_rounds = [first_round]
    st.session_state.current_round = 0
    st.session_state.current_match = 0
    st.session_state.next_round_winners = []
    st.session_state.champion_index = None
    st.session_state.bracket_started = True
    st.session_state.winner_saved = False
    st.session_state.vote_log = []
    st.session_state.current_match_context = None


def advance_until_match_or_winner():
    while True:
        rounds = st.session_state.bracket_rounds
        if not rounds:
            return None, None

        r = st.session_state.current_round
        m = st.session_state.current_match
        round_matches = rounds[r]

        if m >= len(round_matches):
            winners = st.session_state.next_round_winners
            if len(winners) == 1:
                st.session_state.champion_index = winners[0]
                return "winner", winners[0]
            if len(winners) == 0:
                return None, None

            new_round = []
            for i in range(0, len(winners), 2):
                a = winners[i]
                b = winners[i + 1] if i + 1 < len(winners) else None
                new_round.append((a, b))

            st.session_state.bracket_rounds.append(new_round)
            st.session_state.current_round += 1
            st.session_state.current_match = 0
            st.session_state.next_round_winners = []
            continue

        a_idx, b_idx = round_matches[m]

        if a_idx is None and b_idx is None:
            st.session_state.current_match += 1
            continue

        if a_idx is None or b_idx is None:
            winner_idx = b_idx if a_idx is None else a_idx
            st.session_state.next_round_winners.append(winner_idx)
            st.session_state.current_match += 1
            continue

        return "match", (r + 1, m + 1, a_idx, b_idx, len(round_matches))


def entry_image_display(entry):
    """Return something st.image can display."""
    if entry["image_kind"] == "url":
        return entry["image_ref"]  # URL string
    else:
        # local file path
        return str(entry["image_ref"])


def record_vote(choice: str):
    ctx = st.session_state.current_match_context
    if ctx is None:
        return

    round_num = ctx["round_num"]
    match_num = ctx["match_num"]
    a_idx = ctx["a_idx"]
    b_idx = ctx["b_idx"]

    left = st.session_state.entries[a_idx]
    right = st.session_state.entries[b_idx]

    if choice == "left":
        winner_idx, loser_idx = a_idx, b_idx
        winner_side = "left"
    else:
        winner_idx, loser_idx = b_idx, a_idx
        winner_side = "right"

    st.session_state.vote_log.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "round": round_num,
        "match": match_num,

        "left_title": left["title"],
        "left_image_kind": left["image_kind"],
        "left_image_ref": left["image_ref"],

        "right_title": right["title"],
        "right_image_kind": right["image_kind"],
        "right_image_ref": right["image_ref"],

        "winner_side": winner_side,
        "winner_title": st.session_state.entries[winner_idx]["title"],
        "loser_title": st.session_state.entries[loser_idx]["title"],
    })

    st.session_state.next_round_winners.append(winner_idx)
    st.session_state.current_match += 1
    st.session_state.current_match_context = None
    st.rerun()


def save_current_bracket_to_history():
    if st.session_state.winner_saved:
        return
    winner_idx = st.session_state.champion_index
    if winner_idx is None:
        return

    winner = st.session_state.entries[winner_idx]
    record = {
        "id": str(uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "winner_title": winner["title"],
        "winner_image_kind": winner["image_kind"],
        "winner_image_ref": winner["image_ref"],
        "entries": st.session_state.entries,
        "rounds": st.session_state.bracket_rounds,
        "vote_log": st.session_state.vote_log,
    }
    st.session_state.history.append(record)
    save_history(st.session_state.history)
    st.session_state.winner_saved = True


# ---------------- UI ----------------

def page_current_bracket():
    st.header("Create & Run Bracket")

    # # Add entries (URL or Upload)
    # st.subheader("Add entries")

    # if st.session_state.bracket_started:
    #     st.info("Bracket already started — reset bracket to edit entries.")
    # else:
    #     with st.form("add_entry_form"):
    #         title = st.text_input("Description / Name")

    #         mode = st.radio("Image source", ["URL", "Upload"], horizontal=True)

    #         image_url = ""
    #         uploaded_file = None

    #         if mode == "URL":
    #             image_url = st.text_input("Image URL (https://...)")
    #             # Preview (best-effort)
    #             if image_url.strip():
    #                 st.caption("Preview:")
    #                 image_preview(image_url.strip(), max_width=200)
    #         else:
    #             uploaded_file = st.file_uploader(
    #                 "Upload image (png/jpg/webp)",
    #                 type=["png", "jpg", "jpeg", "webp"],
    #             )
    #             if uploaded_file is not None:
    #                 st.caption("Preview:")
    #                 image_preview(uploaded_file, max_width=200)

    #         submitted = st.form_submit_button("Add entry")

    #         if submitted:
    #             if len(st.session_state.entries) >= MAX_ENTRIES:
    #                 st.warning(f"Max of {MAX_ENTRIES} entries reached.")
    #             elif not title.strip():
    #                 st.warning("Description / Name is required.")
    #             elif mode == "URL" and not image_url.strip():
    #                 st.warning("Image URL is required.")
    #             elif mode == "Upload" and uploaded_file is None:
    #                 st.warning("Please upload an image.")
    #             else:
    #                 if mode == "URL":
    #                     image_kind = "url"
    #                     image_ref = image_url.strip()
    #                 else:
    #                     # Save uploaded image to disk so it persists and can appear in history
    #                     suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    #                     fname = f"{uuid4().hex}{suffix}"
    #                     out_path = UPLOAD_DIR / fname
    #                     out_path.write_bytes(uploaded_file.getbuffer())
    #                     image_kind = "file"
    #                     image_ref = str(out_path)

    #                 st.session_state.entries.append({
    #                     "id": len(st.session_state.entries),
    #                     "title": title.strip(),
    #                     "image_kind": image_kind,
    #                     "image_ref": image_ref,
    #                 })
    #                 st.success(f"Added: {title.strip()}")
    #                 st.rerun()

    st.subheader("Add entries")

    if st.session_state.bracket_started:
        st.info("Bracket already started — reset bracket to edit entries.")
    else:
        # ---- Inputs (NOT in a form) so they rerun live ----
        mode = st.radio("Image source", ["URL", "Upload"], horizontal=True, key="image_mode")

        title = st.text_input("Description / Name", key="title_input")

        image_url = ""
        uploaded_file = None

        if mode == "URL":
            image_url = st.text_input("Image URL (https://...)", key="image_url_input")

            # ✅ live preview as you type
            if image_url.strip():
                st.caption("Preview:")
                try:
                    st.image(image_url.strip(), width=200)
                except Exception:
                    st.warning("Could not preview that URL (might be invalid or blocked).")

        else:
            uploaded_file = st.file_uploader(
                "Upload image (png/jpg/webp)",
                type=["png", "jpg", "jpeg", "webp"],
                key="upload_input"
            )

            # ✅ live preview as soon as you pick a file
            if uploaded_file is not None:
                st.caption("Preview:")
                st.image(uploaded_file, width=200)

        # ---- Add button ----
        add_disabled = len(st.session_state.entries) >= MAX_ENTRIES
        if st.button("Add entry", disabled=add_disabled):
            if len(st.session_state.entries) >= MAX_ENTRIES:
                st.warning(f"Max of {MAX_ENTRIES} entries reached.")
            elif not title.strip():
                st.warning("Description / Name is required.")
            elif mode == "URL" and not image_url.strip():
                st.warning("Image URL is required.")
            elif mode == "Upload" and uploaded_file is None:
                st.warning("Please upload an image.")
            else:
                if mode == "URL":
                    image_kind = "url"
                    image_ref = image_url.strip()
                else:
                    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
                    fname = f"{uuid4().hex}{suffix}"
                    out_path = UPLOAD_DIR / fname
                    out_path.write_bytes(uploaded_file.getbuffer())
                    image_kind = "file"
                    image_ref = str(out_path)

                st.session_state.entries.append({
                    "id": len(st.session_state.entries),
                    "title": title.strip(),
                    "image_kind": image_kind,
                    "image_ref": image_ref,
                })

                # clear inputs after add (similar to clear_on_submit)
                st.session_state.title_input = ""
                st.session_state.image_url_input = ""
                st.session_state.upload_input = None

                st.success(f"Added: {title.strip()}")
                st.rerun()


    st.write(f"Current entries: **{len(st.session_state.entries)} / {MAX_ENTRIES}**")

    # Entry list w/ thumbnail + remove buttons
    if st.session_state.entries:
        with st.expander("Entries (preview + remove)"):
            for i, e in enumerate(st.session_state.entries):
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    image_preview(entry_image_display(e), max_width=120)
                with cols[1]:
                    st.markdown(f"**#{i+1}: {e['title']}**")
                    st.caption(f"{e['image_kind']}: {e['image_ref']}")
                with cols[2]:
                    if st.button("Remove", key=f"remove_{i}", disabled=st.session_state.bracket_started):
                        # Optionally delete local file
                        if e["image_kind"] == "file":
                            try:
                                Path(e["image_ref"]).unlink(missing_ok=True)
                            except Exception:
                                pass
                        st.session_state.entries.pop(i)
                        # reassign ids
                        for j, ent in enumerate(st.session_state.entries):
                            ent["id"] = j
                        st.rerun()

    col_start, col_reset = st.columns(2)
    with col_start:
        if st.button("Randomize & start bracket", disabled=(len(st.session_state.entries) < 2 or st.session_state.bracket_started)):
            generate_bracket()
            st.rerun()
    with col_reset:
        if st.button("Reset all (clear entries & bracket)"):
            reset_everything()
            st.rerun()

    st.markdown("---")

    if not st.session_state.bracket_started:
        st.info("Add entries and click **Randomize & start bracket** to begin.")
        return

    status, data = advance_until_match_or_winner()

    if status == "winner":
        winner_idx = data
        winner = st.session_state.entries[winner_idx]

        st.success("Tournament complete!")
        st.subheader("🏆 Winner")
        st.image(entry_image_display(winner), caption=winner["title"], use_container_width=True)
        st.write(f"**{winner['title']}** is the champion.")

        save_current_bracket_to_history()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New bracket with same entries"):
                reset_bracket_only()
                st.rerun()
        with c2:
            if st.button("Reset everything (new entries)"):
                reset_everything()
                st.rerun()

        if st.session_state.vote_log:
            st.markdown("### This bracket's vote log")
            st.dataframe(pd.DataFrame(st.session_state.vote_log), use_container_width=True)

    elif status == "match":
        round_num, match_num, a_idx, b_idx, total_matches = data
        left = st.session_state.entries[a_idx]
        right = st.session_state.entries[b_idx]

        st.session_state.current_match_context = {
            "round_num": round_num,
            "match_num": match_num,
            "a_idx": a_idx,
            "b_idx": b_idx,
            "total_matches": total_matches,
        }

        st.subheader(f"Round {round_num} – Match {match_num} / {total_matches}")
        colL, colR = st.columns([1, 1])

        with colL:
            image_vote(entry_image_display(left), max_width=350)
            st.markdown(f"**{left['title']}**")
            if st.button("Winner", key=f"left_{round_num}_{match_num}"):
                record_vote("left")

        with colR:
            image_vote(entry_image_display(right), max_width=350)
            st.markdown(f"**{right['title']}**")
            if st.button("Winner ", key=f"right_{round_num}_{match_num}"):
                record_vote("right")

    else:
        st.warning("Bracket state looks weird. Try resetting & restarting.")


def page_history():
    st.header("Past Brackets")

    history = st.session_state.history
    if not history:
        st.info("No saved brackets yet. Finish a bracket to save it here.")
        return

    history_sorted = sorted(history, key=lambda r: r["created_at"], reverse=True)

    labels = [
        f"{i+1}. {h['winner_title']} (created {h['created_at']})"
        for i, h in enumerate(history_sorted)
    ]
    idx = st.selectbox(
        "Select a past bracket",
        options=list(range(len(history_sorted))),
        format_func=lambda i: labels[i],
        key="history_select",
    )

    record = history_sorted[idx]

    st.subheader("Winner")
    winner_entry = {
        "title": record["winner_title"],
        "image_kind": record["winner_image_kind"],
        "image_ref": record["winner_image_ref"],
    }
    st.image(entry_image_display(winner_entry), caption=winner_entry["title"], use_container_width=True)
    st.caption(f"Created at: {record['created_at']}")

    vote_log = record.get("vote_log", [])
    st.markdown("### Vote history (every step)")
    if vote_log:
        df = pd.DataFrame(vote_log)
        st.dataframe(df, use_container_width=True)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download vote history as CSV",
            data=csv_buf.getvalue(),
            file_name=f"bracket_{record['id']}_votes.csv",
            mime="text/csv",
        )
    else:
        st.info("No vote history stored for this bracket.")


def main():
    ensure_state()
    st.set_page_config(page_title="Picture Bracket", page_icon="🏆", layout="wide")
    st.title("🏆 Picture Bracket")

    tab_current, tab_history = st.tabs(["Current Bracket", "Bracket History"])
    with tab_current:
        page_current_bracket()
    with tab_history:
        page_history()


if __name__ == "__main__":
    main()