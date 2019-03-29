import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir("./face"): 
        dst ="face_" + str(i) + ".jpg"
        src ='./face/'+ filename 
        dst ='./face/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 