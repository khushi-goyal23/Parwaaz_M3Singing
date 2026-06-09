import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import os
import json
import base64
import streamlit.components.v1 as components

from analyzer import analyze_pitch  # your backend

st.title("🎤 Singing Evaluation System")


def render_live_note_visualizer(audio_path, note_timeline):
    if not note_timeline:
        st.info("No note timeline detected for live view.")
        return

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    note_json = json.dumps(note_timeline)
    unique_notes = []
    for seg in note_timeline:
        n = seg.get("note")
        if n and n != "Rest" and n not in unique_notes:
            unique_notes.append(n)
    unique_notes_json = json.dumps(unique_notes)
    ext = os.path.splitext(audio_path)[1].lower()
    mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
    html = f"""
    <div style="background:#0c1220;border:1px solid #2a324a;border-radius:12px;padding:14px;">
      <audio id="liveAudio" controls style="width:100%;margin-bottom:12px;">
        <source src="data:{mime};base64,{audio_b64}" type="{mime}" />
        Your browser does not support audio playback.
      </audio>
      <div id="nowNote" style="font-size:1.2rem;font-weight:600;margin-bottom:8px;">Current note: --</div>
      <div id="chips" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;"></div>
      <div style="font-size:0.9rem;color:#9fb0d6;margin-top:4px;">
        Notes used in this clip (shown once): highlighted live during playback.
      </div>
    </div>
    <script>
      const notes = {note_json};
      const uniqueNotes = {unique_notes_json};
      const audio = document.getElementById("liveAudio");
      const nowNote = document.getElementById("nowNote");
      const chips = document.getElementById("chips");

      uniqueNotes.forEach((n) => {{
        const chip = document.createElement("span");
        chip.id = "note-chip-" + n.replace("#", "sharp");
        chip.textContent = n;
        chip.style.cssText = "padding:7px 14px;border-radius:999px;background:#1d2438;color:#cfd7f2;font-size:1rem;font-weight:600;";
        chips.appendChild(chip);
      }});

      function tick() {{
        const t = audio.currentTime || 0;
        let active = -1;
        for (let i = 0; i < notes.length; i++) {{
          if (t >= notes[i].start && t < notes[i].end) {{
            active = i;
            break;
          }}
        }}

        uniqueNotes.forEach((n) => {{
          const chip = document.getElementById("note-chip-" + n.replace("#", "sharp"));
          if (!chip) return;
          chip.style.background = "#1d2438";
          chip.style.color = "#cfd7f2";
        }});

        if (active >= 0) {{
          if (notes[active].note === "Rest") {{
            nowNote.textContent = "Current note: (rest / unvoiced)";
          }} else {{
            nowNote.textContent = "Current note: " + notes[active].note;
            const id = "note-chip-" + notes[active].note.replace("#", "sharp");
            const chip = document.getElementById(id);
            if (chip) {{
              chip.style.background = "#2f7dff";
              chip.style.color = "#ffffff";
            }}
          }}
        }} else {{
          nowNote.textContent = "Current note: (rest / unvoiced)";
        }}
      }}

      audio.addEventListener("timeupdate", tick);
      audio.addEventListener("seeked", tick);
      audio.addEventListener("play", tick);
      audio.addEventListener("pause", tick);
      tick();
    </script>
    """
    components.html(html, height=260)


def get_note_summary(note_timeline):
    summary = {}
    for seg in note_timeline:
        note = seg.get("note")
        if note == "Rest":
            continue
        duration = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        summary[note] = summary.get(note, 0.0) + duration
    ordered = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    return [{"note": n, "total_seconds": round(d, 2)} for n, d in ordered]

# Upload files
ref_file = st.file_uploader("Upload Reference Audio", type=["mp3", "wav"])
user_file = st.file_uploader("Upload Your Singing", type=["mp3", "wav"])

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_user_audio_path" not in st.session_state:
    st.session_state.last_user_audio_path = None

if ref_file and user_file:

    # Save temp files
    ref_ext = os.path.splitext(ref_file.name)[1] or ".wav"
    user_ext = os.path.splitext(user_file.name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ref_ext) as f:
        f.write(ref_file.read())
        ref_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=user_ext) as f:
        f.write(user_file.read())
        user_path = f.name

    if st.button("Analyze"):

        with st.spinner("Analyzing... (first run can take time due to model loading)"):
            try:
                result = analyze_pitch(user_path, ref_path)
                st.session_state.analysis_result = result
                st.session_state.last_user_audio_path = user_path
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    user_path = st.session_state.last_user_audio_path
    st.success("Analysis Complete")

    # 🎵 AUDIO
    st.subheader("Your Audio")
    st.audio(user_path)
    st.subheader("Live Note Tracking")
    note_timeline = result.get("note_timeline", [])
    render_live_note_visualizer(user_path, note_timeline)
    st.subheader("Note Summary")
    st.write(get_note_summary(note_timeline))

    # 📊 METRICS
    st.subheader("Scores")
    st.write(f"Pitch Accuracy: {result['in_range_percent']:.2f}%")
    st.write(f"Final Score: {result['final_score']:.2f}")

    # 🎼 SCALE
    st.subheader("Detected Scale")
    st.write(result["detected_scale"])
    st.caption("Overall estimated key/scale from the detected sung notes across the clip.")

    # 🔁 TRANSITIONS
    st.subheader("Note Transitions")
    st.write(result["note_transitions"])

    # ⏱ DURATIONS
    st.subheader("Note Durations")
    st.write(result["note_durations"])

    # 📈 PLOT
    st.subheader("Pitch Contour")

    pitch = result["pitch_contour"]
    fig, ax = plt.subplots()
    ax.plot(pitch)
    ax.set_title("Pitch over Time")
    st.pyplot(fig)

    # 📍 TIMELINE FEEDBACK
    st.subheader("Timeline Feedback")

    for seg in result["timeline_feedback"]:
        st.write(f"{seg['start']}s - {seg['end']}s: {seg['message']}")

    # 💬 FEEDBACK
    st.subheader("Overall Feedback")
    st.write(result["feedback"])