from email_spam_filter import classify

email = "Congratulations! You have won a free vacation."

result = classify(email)

print("Prediction:", result.label)
print("Spam probability:", result.spam_probability)