"""Streamlit interface for adapting and sampling the local commentary model."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from app.core import MatchEvent, format_prompt, seed_examples
from app.llm import TrainedModel, generate, train_lora
from app.storage import append_record


@st.cache_resource(show_spinner=False)
def cached_train(steps: int) -> TrainedModel:
    return train_lora(steps=steps)


def main() -> None:
    st.set_page_config(page_title="Sports Commentary Generator", page_icon="🏟️", layout="wide")
    st.title("🏟️ Sports Commentary Generator")
    st.caption("Build a lightweight LoRA adapter from scratch and turn match events into live-sounding calls.")
    with st.sidebar:
        st.header("Adapter training")
        steps = st.slider("Training steps", 40, 500, 160, 20)
        st.markdown("The base projection stays frozen; only the low-rank A and B matrices are trained.")
        if st.button("Train / load adapter", type="primary"):
            with st.spinner("Training the local PyTorch adapter…"):
                st.session_state.adapter_model = cached_train(steps)
            st.success(f"Adapter ready · final batch loss {st.session_state.adapter_model.training_loss:.3f}")
        if "adapter_model" not in st.session_state:
            st.info("Train the adapter to enable generation.")
        st.divider()
        st.caption("No model download or external API is used.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Match event")
        with st.form("event_form"):
            sport = st.selectbox("Sport", ["Soccer", "Basketball", "Hockey", "American Football", "Tennis", "Other"])
            event_type = st.selectbox("Event", ["Goal", "Score change", "Foul", "Red card", "Buzzer-beater", "Penalty saved", "Touchdown", "Match point", "Power play"])
            col1, col2 = st.columns(2)
            team = col1.text_input("Team", "Home United")
            player = col2.text_input("Player", "Alex Morgan")
            opponent = col1.text_input("Opponent", "City Rovers")
            score = col2.text_input("Score", "2-1")
            time = col1.text_input("Match time", "87th minute")
            context = st.text_area("Situation", "A tense late-game finish")
            submitted = st.form_submit_button("Generate commentary", type="primary")
    with right:
        st.subheader("Live call")
        if submitted:
            try:
                prompt = format_prompt(MatchEvent(sport, event_type, team, player, opponent, score, time, context))
            except ValueError as exc:
                st.error(str(exc))
            else:
                model = st.session_state.get("adapter_model")
                if model is None:
                    st.warning("Train the adapter in the sidebar first.")
                else:
                    with st.spinner("Calling the play…"):
                        output = generate(model, prompt, temperature=0.72)
                    st.markdown(f"> {output}")
                    append_record("data/generations.jsonl", {"timestamp": datetime.now(timezone.utc).isoformat(), "prompt": prompt, "commentary": output})
        st.markdown("**Seed examples**")
        st.dataframe([{"event": p, "commentary": t} for p, t in seed_examples()[:5]], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

