import joblib
import json
import streamlit as st
import random
import datetime
import os
import csv

#Loads the trained model and vectorizer

clf=joblib.load("chatbot_model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

# Load the dataset
with open("Dataset.json", "r") as file:
    data = json.load(file)

# Chatbot function
def chatbot(input_text):
    # Normalize input text
    input_text = input_text.lower().strip()
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    
    # Find the corresponding response
    for intent in data:
        if intent["tag"] == tag:
            responses = intent["responses"]
            return random.choice(responses)
    return "I'm sorry, I didn't understand that."    

def main():
    st.title("Intents Based Chatbot using NLP")

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Sidebar menu
    menu = ["Home", "Conversations History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    #Displaying home
    if choice == "Home":
        st.write("Welcome to ChatEase. Your seamless conversation companion.")
        
        # Create a form to group the text input and button together
        with st.form(key="chat_form"):
            # User input
            user_input = st.text_input("You:", key="user_input")

            # Add a submit button inside the form
            submit_button = st.form_submit_button("Send")

            # If the submit button is clicked, process the input
            if submit_button and user_input.strip():
                response = chatbot(user_input)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Save conversation to session state
                st.session_state["history"].append((user_input, response, timestamp))

                # Display chatbot response
                st.write(f"**Chatbot**: {response}")

                # Save to CSV
                if not os.path.exists('chat_log.csv'):
                    with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])

                if response.lower() in ['goodbye', 'bye']:
                    st.write("Thank you for chatting with me. Have a great day!")
                    st.stop()

    

    # Conversations History section
    elif choice == "Conversations History":
        st.header("Conversations History")
        
        # Check if there is conversation history in the session state
        if st.session_state["history"]:
            # Create a DataFrame for tabular display
            import pandas as pd
            history_df = pd.DataFrame(
                st.session_state["history"],
                columns=["User Input", "Chatbot Response", "Timestamp"]
            )
            # Display the DataFrame
            st.dataframe(history_df)
        else:
            st.write("No conversation history available.")

    # About section
    elif choice == "About":
        # Title for the About Page
        st.title("About This Chatbot")

        # Brief description
        st.write("""
        This intents-based chatbot is designed to understand user queries, classify them into predefined categories (intents), 
        and provide relevant responses. Built using **Natural Language Processing (NLP)** techniques and **Logistic Regression**, 
        the chatbot offers an interactive experience to users.
        """)

        # Features of the Chatbot
        st.subheader("Features:")
        st.write("""
        - **Intent Classification**: The chatbot classifies user inputs into categories such as greeting, farewell, etc.
        - **Natural Language Processing (NLP)**: The chatbot uses advanced text preprocessing techniques like tokenization and TF-IDF.
        - **Streamlit Interface**: Provides a clean, user-friendly interface for seamless interaction.
        - **Real-Time Responses**: Instant replies based on user inputs.
        """)

        # Tools and Technologies Used
        st.subheader("Tools and Technologies Used:")
        st.write("""
        - **Programming Language**: Python  
        - **Libraries**:  
        - `scikit-learn` for Logistic Regression  
        - `nltk` and `json` for NLP and data processing  
        - `Streamlit` for the web interface  
        - **Environment**: Google Colab for development and testing
        """)

        # Future Enhancements
        st.subheader("Future Enhancements:")
        st.write("""
        - Integration with a database to save user conversations.
        - Implementation of more advanced models like transformers for better intent recognition.
        - Adding support for multiple languages.
        """)

        # Footer
        st.write("**Developed by:** Nishant Kumar Bhadani")

# Run the app
if __name__ == "__main__":
    main()
