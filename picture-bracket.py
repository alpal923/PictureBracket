import streamlit as st
import random
import json
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import io
import pandas as pd

# ------------- Config -------------

MAX_ENTRIES = 64
HISTORY_FILE = Path("bracket_history.json")

# ------------- Persistence helpers -------------

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ------------- State init -------------

def ensure_state():
    defaults = {
        "entries": [],                  # list of {id, title, image_url}
        "bracket_rounds": [],           # list of rounds; each is list of (idx_a, idx_b)
        "current_round": 0,
        "current_match": 0,
        "next_round_winners": [],
        "champion_index": None,
        "bracket_started": False,
        "winner_saved": False,
        "history": load_history(),
        "vote_log": [],                 # list of vote events for current bracket
        "current_match_context": None,  # details of current matchup
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ------------- Bracket logic -------------

def generate_bracket():
    """Build the first round from entries, padded to a power of two with byes."""
    entries = st.session_state.entries
    n = len(entries)
    if n < 2:
        return

    indices = list(range(n))
    random.shuffle(indices)

    # Pad to next power of 2 with None (byes)
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
    """
    Skip bye matches and build new rounds as needed.

    Returns:
      ("winner", winner_idx) OR
      ("match", (round_num, match_num, idx_left, idx_right, total_matches))
    """
    while True:
        rounds = st.session_state.bracket_rounds
        if not rounds:
            return None, None

        r = st.session_state.current_round
        m = st.session_state.current_match
        round_matches = rounds[r]

        # Finished this round -> build next or declare champion
        if m >= len(round_matches):
            winners = st.session_state.next_round_winners
            if len(winners) == 1:
                st.session_state.champion_index = winners[0]
                return "winner", winners[0]
            elif len(winners) == 0:
                return None, None
            else:
                new_round = []
                for i in range(0, len(winners), 2):
                    a = winners[i]
                    b = winners[i + 1] if i + 1 < len(winners) else None
                    new_round.append((a, b))
                st.session_state.bracket_rounds.append(new_round)
                st.session_state.current_round += 1
                st.session_state.current_match = 0
                st.session_state.next_round_winners = []
                continue  # evaluate new round

        # Examine current match
        a_idx, b_idx = round_matches[m]

        # Both are byes
        if a_idx is None and b_idx is None:
            st.session_state.current_match += 1
            continue

        # One is a bye -> auto-advance
        if a_idx is None or b_idx is None:
            winner_idx = b_idx if a_idx is None else a_idx
            st.session_state.next_round_winners.append(winner_idx)
            st.session_state.current_match += 1
            continue

        # Real matchup
        round_num = r + 1
        match_num = m + 1
        total_matches = len(round_matches)
        return "match", (round_num, match_num, a_idx, b_idx, total_matches)


def record_vote(choice: str):
    """choice is 'left' or 'right' for the current match."""
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
        winner_idx = a_idx
        winner_side = "left"
        loser_idx = b_idx
    else:
        winner_idx = b_idx
        winner_side = "right"
        loser_idx = a_idx

    # Append vote event to log
    st.session_state.vote_log.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "round": round_num,
            "match": match_num,
            "left_index": a_idx,
            "left_title": left["title"],
            "left_image_url": left["image_url"],
            "right_index": b_idx,
            "right_title": right["title"],
            "right_image_url": right["image_url"],
            "winner_side": winner_side,
            "winner_index": winner_idx,
            "winner_title": st.session_state.entries[winner_idx]["title"],
            "loser_index": loser_idx,
            "loser_title": st.session_state.entries[loser_idx]["title"],
        }
    )

    # Advance bracket state
    r = st.session_state.current_round
    m = st.session_state.current_match
    st.session_state.next_round_winners.append(winner_idx)
    st.session_state.current_match += 1
    st.session_state.current_match_context = None

    st.experimental_rerun()


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


def save_current_bracket_to_history():
    # Guard: only save once
    if st.session_state.winner_saved:
        return

    winner_idx = st.session_state.champion_index
    if winner_idx is None:
        return

    winner = st.session_state.entries[winner_idx]
    record = {
        "id": str(uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "winner_index": winner_idx,
        "winner_title": winner["title"],
        "winner_image_url": winner["image_url"],
        "entries": st.session_state.entries,
        "rounds": st.session_state.bracket_rounds,
        "vote_log": st.session_state.vote_log,  # FULL step-by-step history
    }
    st.session_state.history.append(record)
    save_history(st.session_state.history)
    st.session_state.winner_saved = True


# ------------- UI: Current Bracket -------------

def page_current_bracket():
    st.header("Create & Run Bracket")

    # Entry form
    with st.form("add_entry_form"):
        st.subheader("Add entries")
        col1, col2 = st.columns([2, 3])
        with col1:
            title = st.text_input("Description / Name")
        with col2:
            image_url = st.text_input("Image URL (https://...)")

        submitted = st.form_submit_button("Add entry")
        if submitted:
            if len(st.session_state.entries) >= MAX_ENTRIES:
                st.warning(f"Max of {MAX_ENTRIES} entries reached.")
            elif not title or not image_url:
                st.warning("Both description and image URL are required.")
            else:
                st.session_state.entries.append(
                    {
                        "id": len(st.session_state.entries),
                        "title": title.strip(),
                        "image_url": image_url.strip(),
                    }
                )
                st.success(f"Added: {title.strip()}")

    st.write(f"Current entries: **{len(st.session_state.entries)} / {MAX_ENTRIES}**")

    if st.session_state.entries:
        with st.expander("Show entries"):
            for i, e in enumerate(st.session_state.entries, start=1):
                st.markdown(f"**#{i}** – {e['title']}")
                st.caption(e["image_url"])

    col_start, col_reset = st.columns(2)
    with col_start:
        if st.button("Randomize & start bracket", disabled=len(st.session_state.entries) < 2):
            if len(st.session_state.entries) < 2:
                st.warning("Need at least 2 entries.")
            else:
                generate_bracket()
                st.experimental_rerun()
    with col_reset:
        if st.button("Reset all (clear entries & bracket)"):
            reset_everything()
            st.experimental_rerun()

    st.markdown("---")

    # Bracket / match UI
    if not st.session_state.bracket_started:
        st.info("Add entries and click **Randomize & start bracket** to begin.")
        return

    status, data = advance_until_match_or_winner()

    if status == "winner":
        winner_idx = data
        winner = st.session_state.entries[winner_idx]

        st.success("Tournament complete!")
        st.subheader("🏆 Winner")

        st.image(winner["image_url"], caption=winner["title"], use_column_width=True)
        st.write(f"**{winner['title']}** is the champion.")

        # Save to history (if not done already)
        save_current_bracket_to_history()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New bracket with same entries"):
                reset_bracket_only()
                st.experimental_rerun()
        with col2:
            if st.button("Reset everything (new entries)"):
                reset_everything()
                st.experimental_rerun()

        # Show the vote log for this just-completed bracket
        if st.session_state.vote_log:
            st.markdown("### This bracket's vote log")
            df = pd.DataFrame(st.session_state.vote_log)
            st.dataframe(df, use_container_width=True)

    elif status == "match":
        round_num, match_num, a_idx, b_idx, total_matches = data
        left = st.session_state.entries[a_idx]
        right = st.session_state.entries[b_idx]

        # Store context so record_vote can log properly
        st.session_state.current_match_context = {
            "round_num": round_num,
            "match_num": match_num,
            "a_idx": a_idx,
            "b_idx": b_idx,
            "total_matches": total_matches,
        }

        st.subheader(f"Round {round_num} – Match {match_num} / {total_matches}")

        colL, colR = st.columns(2)

        with colL:
            st.image(left["image_url"], use_column_width=True)
            st.markdown(f"**{left['title']}**")
            if st.button("Winner", key=f"left_{round_num}_{match_num}"):
                record_vote("left")

        with colR:
            st.image(right["image_url"], use_column_width=True)
            st.markdown(f"**{right['title']}**")
            if st.button("Winner ", key=f"right_{round_num}_{match_num}"):
                record_vote("right")

    else:
        st.warning("Bracket state looks weird. Try resetting & restarting.")


# ------------- UI: History -------------

def page_history():
    st.header("Past Brackets")

    history = st.session_state.history
    if not history:
        st.info("No saved brackets yet. Finish a bracket to save it here.")
        return

    # Sort newest first
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
    st.image(record["winner_image_url"], caption=record["winner_title"], use_column_width=True)
    st.caption(f"Created at: {record['created_at']}")

    with st.expander("Entries in this bracket"):
        for i, e in enumerate(record["entries"], start=1):
            st.markdown(f"**#{i}** – {e['title']}")
            st.caption(e["image_url"])

    # Vote log (step-by-step history)
    vote_log = record.get("vote_log", [])
    st.markdown("### Vote history for this bracket")
    if vote_log:
        df = pd.DataFrame(vote_log)
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download vote history as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"bracket_{record['id']}_votes.csv",
            mime="text/csv",
        )
    else:
        st.info("No vote history stored for this bracket (probably from an older version of the app).")


# ------------- Main -------------

def main():
    ensure_state()
    st.set_page_config(page_title="Bracket Vote", page_icon="🏆", layout="wide")

    st.title("🏆 Bracket Vote App")

    tab_current, tab_history = st.tabs(["Current Bracket", "Bracket History"])

    with tab_current:
        page_current_bracket()
    with tab_history:
        page_history()


if __name__ == "__main__":
    main()