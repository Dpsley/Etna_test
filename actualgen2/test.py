from catboost import Pool, CatBoostClassifier

train_data = [["France", 1924, 44],
              ["USA", 1932, 37],
              ["USA", 1980, 37]]

eval_data = [["USA", 1996, 197],
             ["France", 1968, 37],
             ["USA", 2002, 77]]

cat_features = [0]

train_label = [1, 1, 0]
eval_label = [0, 0, 1]

train_dataset = Pool(data=train_data,
                     label=train_label,
                     cat_features=cat_features)

eval_dataset = Pool(data=eval_data,
                    label=eval_label,
                    cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=100)
# Fit model with `use_best_model=True`

model.fit(train_dataset,
          use_best_model=True,
          eval_set=eval_dataset)

print("Count of trees in model = {}".format(model.tree_count_))
model.save_model("catboost_model.cbm")
print("✅ модель сохранена в catboost_model2.cbm")