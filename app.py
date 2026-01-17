
## **10. Web Interface with Gradio (10 Marks)**
#Create a user-friendly Gradio web interface that takes user inputs and displays the prediction from your trained model.


import gradio as gr
import pandas as pd
import pickle


# Load the saved model
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

def predict_loan(income, credit_score, loan_amount, years_employed, points):
    input_data = pd.DataFrame({
        'income': [income],
        'credit_score': [credit_score],
        'loan_amount': [loan_amount],
        'years_employed': [years_employed],
        'points': [points],
        'income_to_loan_ratio': [income / (loan_amount + 1)]
    })
    prediction = loaded_model.predict(input_data)[0]
    return "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Income"),
        gr.Number(label="Credit Score"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Years Employed"),
        gr.Number(label="Points")
    ],
    outputs="text",
    title="Loan Approval Prediction System",
    description="Enter your details to check if your loan is likely to be approved."
)

demo.launch(share = True)

"""## **11. Deployment to Hugging Face (10 Marks)**
Deploy the Gradio app to Hugging Face Spaces and ensure it is accessible via a public URL.
"""

