def loss(target,out_image):
    criterion = nn.MSELoss()
    weights = [0.005, 0.01, 0.02, 0.08, 0.32]
    loss = 0
    for i in range(5):
        loss+= weights[i]*criterion(target,out_image[i])
    return loss
