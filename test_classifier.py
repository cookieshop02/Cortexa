from classifier import classify_text

sample_text = "User complained yesterday about bad delivery. User generally prefers spicy food."

result = classify_text(sample_text)

for unit in result:
    print(unit)
    print("---")

