from storage import store_triplet

# Pehle: Mumbai wala fact store karo
result1 = store_triplet(
    user_id="user_1",
    entity1="User",
    relationship="lives_in",
    entity2="Mumbai",
    confidence=0.9,
    importance_category="high",
    importance_score=0.8
)
print("First store:", result1)

# Ab: Bangalore wala fact store karo — Mumbai wala automatically invalidate hona chahiye
result2 = store_triplet(
    user_id="user_1",
    entity1="User",
    relationship="lives_in",
    entity2="Bangalore",
    confidence=0.9,
    importance_category="high",
    importance_score=0.8
)
print("Second store:", result2)