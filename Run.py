import Test
import Train_Face
import Training

print("Press 1 to start capturing a new face")
print("Press 2 to train the face")
print("Press 3 for the recognition and identification of face")
print("Press 4 to Exit the program")
while True:
    ch = int(input("Enter your option: "))
    if ch==1:
        Training.Training()
    elif ch==2:
        Train_Face.Train()
    elif ch==3:
        Test.Test()
    else:
        exit()

