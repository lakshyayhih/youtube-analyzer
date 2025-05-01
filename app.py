import streamlit as st
from utils import (
    extract_video_id,
    get_transcript,
    get_named_entities,
    summarize_text,
    analyze_sentiment,
    sentiment_over_time,
    generate_wordcloud,
    extract_keywords,
    create_timeline
)

# Streamlit App
st.set_page_config(page_title="🎥 YouTube Transcript Analyzer", layout="wide")
st.title("🎥 YouTube Transcript Analyzer")

# URL Input
url = st.text_input("Paste a YouTube video URL below:")

# Proceed if URL is provided
if url:
    with st.spinner("🔍 Analyzing..."):
        try:
            # 1. Extract video ID and get transcript
            video_id = extract_video_id(url)
            transcript = get_transcript(video_id)

            # 2. Transcript Preview
            st.subheader("📄 Transcript Preview")
            st.write(transcript[:1000] + "...")

            # 3. Named Entity Recognition
            """entities = get_named_entities(transcript)
            st.subheader("🧑‍🤝‍🧑 Named Entities")
            st.write(entities)"""

            # 4. Summary
            summary = summarize_text(transcript)
            st.subheader("🧠 Summary")
            st.success(summary)

            # 5. Overall Sentiment
            sentiment = analyze_sentiment(transcript)
            st.subheader("📊 Overall Sentiment")
            st.info(f"Sentiment: **{sentiment}**")

            # 6. Sentiment Over Time
            sentiments = sentiment_over_time(transcript)
            st.subheader("📈 Sentiment Over Time")
            st.line_chart(sentiments)

            # 7. Word Cloud
            wordcloud = generate_wordcloud(transcript)
            st.subheader("🎨 Word Cloud")
            st.image(wordcloud.to_array())

            # 8. Keywords
            num_keywords = st.slider("🔑 Number of keywords to extract", 5, 30, 10)
            keywords = extract_keywords(transcript, top_n=num_keywords)
            st.subheader("🔑 Top Keywords")
            st.write(", ".join(keywords))

            # 9. Timeline
            st.subheader("⏳ Transcript Timeline (Optional)")
            timeline = create_timeline(transcript)
            st.dataframe(timeline)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
