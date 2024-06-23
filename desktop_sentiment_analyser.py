import tkinter as tk
from tkinter import messagebox
import joblib
import re
import requests

def extract_video_id(url):
    # Define the regular expression pattern for YouTube video ID
    pattern = r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)

    if match:
        return match.group(1)
    else:
        return ""

def get_comments(link):

    videoId = extract_video_id(link)
    if videoId == "":
        print("Error")
        return
    url = "https://youtube-v31.p.rapidapi.com/commentThreads"
    querystring = {"part": "snippet", "videoId": videoId, "maxResults": "100"}

    headers = {
        "x-rapidapi-key": "0891c1fa67mshb75ea0e5d4d19e5p180751jsn5d38bfedeab4",
        "x-rapidapi-host": "youtube-v31.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    responseData = response.json()

    comments = []
    for item in responseData["items"]:
        commentText = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(commentText)

    return comments

def predict(text):
    model = joblib.load('sentiment_analysis_svm_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    return y_pred

def get_sentiment(comments):
    positive = 0
    neutral = 0
    negative = 0
    for comment in comments:
        if predict(comment) == 1:
            positive += 1
        elif predict(comment) == 0:
            neutral += 1
        else:
            negative += 1

    positive = (positive / len(comments)) * 100
    negative = (negative / len(comments)) * 100
    neutral = (neutral / len(comments)) * 100

    return positive, negative, neutral

# Function to handle button click event
def on_predict():
    input_text = text_entry.get("1.0", tk.END).strip()
    if input_text:
        comments = get_comments(input_text)
        positive, negative, neutral = get_sentiment(comments)
        messagebox.showinfo("Prediction", f"Positive sentiment: {positive} \nNegative sentiment: {negative} \nNeutral sentiment: {neutral}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")

# Create and place the text entry widget
text_entry = tk.Text(root, height=3, width=50)
text_entry.pack(padx=10, pady=10)

# Create and place the Predict button
predict_button = tk.Button(root, text="Predict Sentiment", command=on_predict)
predict_button.pack(pady=10)

# Start the main event loop
root.mainloop()
