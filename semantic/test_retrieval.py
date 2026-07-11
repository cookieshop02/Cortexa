from retrieval import search_semantic_memory

result1 = search_semantic_memory("user_1", "Where does the user live?")
print("Current:", result1)

result2 = search_semantic_memory("user_1", "Where did the user used to live?")
print("History:", result2)