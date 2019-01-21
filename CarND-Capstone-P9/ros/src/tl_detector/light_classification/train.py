import model as md

model = md.Model("training_data/labels.txt")
model.run(epochs = 5, batch_size = 16 )
