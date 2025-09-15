import streamlit as st
import pandas as pd
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(page_title="🧘 Yoga Pose Recommender", layout="centered")

# Load dataset
df = pd.read_csv("yoga_pose_data.csv")

# Title
st.title("🧘‍♀️ Yoga Pose Generator Based on Pain Description")

# Language selection
lang = st.selectbox(
    "🌐 Choose Your Language / अपनी भाषा चुनें / మీ భాషను ఎంచుకోండి",
    ["English", "Hindi", "Telugu"]
)

# User fitness level for personalization
fitness_level = st.selectbox(
    "🏋️ Fitness Level",
    ["Beginner", "Intermediate", "Advanced"]
)

# Free-text pain description
pain_input = st.text_input("📝 Describe your pain or discomfort (any language)", "")

# Multilingual pain area mapping lists, if needed for UI
pain_labels = {
    "English": ["Neck", "Shoulders", "Back", "Lower Back", "Hips", "Knees", "Legs", "Spine", "Full Body"],
    "Hindi": ["गर्दन", "कंधे", "पीठ", "निचली पीठ", "कूल्हे", "घुटने", "पैर", "रीढ़", "पूरा शरीर"],
    "Telugu": ["మెడ", "భుజాలు", "వెనుక భాగం", "తక్కువ వెన్ను", "నితంబాలు", "మోకాళ్లు", "కాళ్లు", "స్పైన్", "పూర్తి శరీరం"]
}

lang_map = {
    "गर्दन": "Neck", "कंधे": "Shoulders", "पीठ": "Back", "निचली पीठ": "Lower Back",
    "कूल्हे": "Hips", "घुटने": "Knees", "पैर": "Legs", "रीढ़": "Spine", "पूरा शरीर": "Full Body",
    "మెడ": "Neck", "భుజాలు": "Shoulders", "వెనుక భాగం": "Back", "తక్కువ వెన్ను": "Lower Back",
    "నితంబాలు": "Hips", "మోకాళ్లు": "Knees", "కాళ్లు": "Legs", "స్పైన్": "Spine", "పూర్తి శరీరం": "Full Body"
}

# NLP function to guess closest pain area based on input text
def guess_pain_area(user_text, df):
    choices = df["pain_area"].unique()
    corpus = [user_text] + list(choices)
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    best_idx = sim.argmax()
    return choices[best_idx]

# On search button press
if st.button("🔍 Find Yoga Poses"):
    if pain_input.strip():
        selected_pain = guess_pain_area(pain_input, df)
    else:
        st.warning("❗ Please describe your pain for better recommendations.")
        st.stop()

    # Filter results by pain area and fitness level (assuming difficulty available)
    # For demonstration, we only filter by pain_area
    results = df[(df["pain_area"].str.lower() == selected_pain.lower())]

    if not results.empty:
        st.success(f"✅ Yoga Poses for \"{pain_input}\" (mapped to: {selected_pain}):")
        for index, row in results.iterrows():
            pose_name = row["pose"]
            st.markdown(f"### 🧘 {pose_name}")

            # Show instructions as per language selection
            if lang == "English":
                st.markdown(f"**Instructions:** {row['instructions_english']}")
            elif lang == "Hindi":
                st.markdown(f"**निर्देश:** {row['instructions_hindi']}")
            elif lang == "Telugu":
                st.markdown(f"**సూచనలు:** {row['instructions_telugu']}")

            # Link to Google image search for pose
            query = urllib.parse.quote(pose_name + " yoga pose")
            search_url = f"https://www.google.com/search?tbm=isch&q={query}"
            st.markdown(f"[🖼️ Click for Pose Image]({search_url})", unsafe_allow_html=True)

            # Simple feedback (thumbs up/down)
            st.markdown("**Was this pose helpful?**")
            st.feedback("thumbs", key=f"feedback_{pose_name}_{index}")

            st.markdown("---")
    else:
        st.warning("❌ No yoga poses found for the pain area. Try different description.")