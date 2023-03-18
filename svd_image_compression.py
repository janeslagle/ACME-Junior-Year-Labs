# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #follow Algorithm 1.1 in lab manual: 
    count = 0    #need for step 5
    
    A_H = A.conj().T                        
    eigvalues, V = la.eig(A_H @ A)
    singvalues = np.sqrt(eigvalues)
    
    arg = singvalues.argsort()[::-1]  #this sorts indices in order of greatet to smallest (need to reverse it to get in that order)
    new_singvalues = np.zeros(len(eigvalues))
    new_V = np.zeros((len(V), len(V[0])))
    
    new_singvalues = singvalues[arg]  #put singular values in decreasing order into a new array
    new_V = V[:, arg]                 #put eigenvectors in same order as singular values: arranging them in same order of 
                                      #the decreasing singular values
                                      #go through all rows: when get to argth column: put new_V equal to argth V
    for i in singvalues:
        if i > tol:      #its nonzero if its greater than the tolerance
            count = count + 1
    
    sing_1 = new_singvalues[:count]  #r here is our count (the number of nonzero sing values)
    V_1 = new_V[:,:count]
    
    U_1 = (A @ V_1) / sing_1   #array broadcasting is multiplying arrays: the @
    
    return U_1, sing_1, V_1.conj().T
                                                
   
# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #make 2x200 matrix S:
    S = np.zeros((2,200))   #initialize it with 0s to start out
    #do linspace from 0 to 2pi of our theta values
    #get array of theta values
    
    theta = np.linspace(0, 2 * np.pi, 200)  #gets our theta values
    x = np.cos(theta)                       #our x values on the unit circle
    y = np.sin(theta)                       #our y values on the unit circle
    
    S[0:] = x    #s[0:] says everything in 1st row
    S[1:] = y    #S[1:] says everything in 2nd row
    
    #make the matrix E:
    E = np.array([[1,0,0],[0,0,1]])
    
    #compute the full SVD
    U, sigma, V_H = la.svd(A)  #get SVD, la.svd only makes list of the sigma values, so need to put them into matrix using np.diag
    s = np.diag(sigma)         #matrix of our sigma values
    
    #now plot everything:
    #first subplot: plot S, E
    plt.subplot(2,2,1)
    plt.plot(S[0], S[1])   #S only has 2 rows, so plotting S[0], S[1] will plot all of S
    plt.plot(E[0], E[1])   #plot E in its entirety since E is 2 rows, so plotting both of its rows will plot all of E
    plt.axis("equal")
    
    #second subplot
    plt.subplot(2,2,2)
    plt.plot((V_H @ S)[0], (V_H @ S)[1])   #need to do at 0, 1 so that will plot x,y values since plotting here: need plot x and y
    plt.plot((V_H @ E)[0], (V_H @ E)[1])
    plt.axis("equal")

    #third subplot
    plt.subplot(2,2,3)
    plt.plot((s @ V_H @ S)[0], (s @ V_H @ S)[1])
    plt.plot((s @ V_H @ E)[0], (s @ V_H @ E)[1])
    plt.axis("equal")
    
    #fourth subplot
    plt.subplot(2,2,4)
    plt.plot((U @ s @ V_H @ S)[0], (U @ s @ V_H @ S)[1])
    plt.plot((U @ s @ V_H @ E)[0], (U @ s @ V_H @ E)[1])
    plt.axis("equal") 
    
    plt.show()
    
   
# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #part 1:
    U, sigma, V_H = la.svd(A)  #find compact SVD
    sig = np.diag(sigma)       #put sigma values into matrix (otherwise they are just a list). Need them in matrix to compute A_s
    
    #want truncated SVD: so use splicing to do
    U_s = U[:,:s]       #splicing syntax says: go through all rows, start at beginning of columns and end at the sth column
    sig_s = sig[:s, :s] #get until sth row, sth column
    V_s = V_H[:s, :]    #get until sth row, all columns
    
    A_s = U_s @ sig_s @ V_s   #compute A_s matrix. It is product of all truncated matrices from the SVD of A
    
    #part 2:
    #only want matrix version of sigma when compute A_s, so truncate sigma to be a 1D array. Do this because told in lab manual
    #that in part 2, want sigma be stored as 1D array, not full diagonal matrix
    sigma_array = sigma[:s]   #truncate sigma into a 1D array. Want to truncate with s still. Use this for sigma now
    
    #want number of entries stored in truncated SVD form. So want number of entries in each truncated matrix. 
    #use .size to get number of elements in each matrix
    #the total size want to return sum of the sizes of each matrix
    num_entries = U_s.size + V_s.size + sigma_array.size
    
    #part 3:
    #if s greater than rank(A), then raise ValueError
    r = np.linalg.matrix_rank(A)   #find rank(A)
    if s > r:
        raise ValueError("Rank is less than s")
        
            
    return A_s, num_entries
    
   
# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #part 1:
    #compute compact SVD of A: this is A_s. Find in exact same way found in problem 3
    U, sigma, V_H = la.svd(A)   #get compact SVD
    
    #to find s:
    #want singular value that is less than err. Want the (s+1)th singular value to be less than err
    #look at sig_s list and find whichever sigma is less than err. then find that index of s want, then plug that in for s in problem 3 function
    #because we want the corresponding A_s with that s
    
    sing = np.where(sigma < err, sigma, -1) 
    
    #says: check if every entry in sigma is less than the err. Then the , sigma: says if it is less than the error, then keep it the same
    #and the -1 says: if it is not less than err: make it a -1 so then able to easily see which singular value is the s
    #so then do np.argmax to get the biggest singular value that is less than error because our s here will be the biggest one that satisifies this
    
    s = np.argmax(sing)  #s is the index of the biggest singular value from sigma that is less than err
        
    #part 3:
    #if error value is <= smallest singular value of A, raise ValueError
    
    #if all of the singular values are -1 from our part 1 finding s, then all of the singular values in sigma will be turned to -1
    #so we will only raise the error if all of them are -1
    
    if all ([i == -1 for i in sing]):   #loop through all singular values after checked if less than err and see if they equal -1
        raise ValueError("The error value is less than or equal to the smallest singular value of A")
        
    return svd_approx(A, s)  #will return the matrix and number of entries like want
    
    
# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255  #read image. this is the original image. need divide by 255 here. will make approx 
                                    #also be divided by 255
    og_image_size = image.size      #get the size of the og image since will need for the title of plots make
    
    if len(image.shape) == 3:  #color if its dim is 3
        color = True
    elif len(image.shape) == 2:  #image is grayscale
        color = False
    
    if color == True:
    #use function from problem 3 to find best rank-s approx of image, want A_s from the function
        #separate into the 3 layers
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        
        #get the approximation of those 3 separately. use problem 3 to do this
        R_s = svd_approx(R, s)[0]  #only want A_s here so just return the first thing from problem 3
        G_s = svd_approx(G, s)[0]
        B_s = svd_approx(B, s)[0]
        
        #put all of the approximations together into an array
        approx_image = np.dstack((R_s, G_s, B_s))
        
        #the clipping stuff in lab manual: can just do it in a single line
        approx_image = np.clip(approx_image, 0, 1)   #says take every line in our approx image and the 0 is the min value that you can have, the 1 is max value have
                                                     #have 0 and 1 as the min and max since we want inside interval [0,1]
        #get the size of the approximated image for the plot title will make                                             
        red_size = svd_approx(R,s)[1]
        green_size = svd_approx(G,s)[1]
        blue_size = svd_approx(B,s)[1]
        
        approximated_size = red_size + green_size + blue_size  #get total size of the whole approximated image
        
        size_difference = og_image_size - approximated_size
        
        plt.suptitle("Entries required to store OG image vs approximated image: " + str(size_difference))
        
        #plot OG image 1st:
        plt.subplot(1,2,1)
        plt.imshow(image, cmap = "gray")
        plt.title("Original Image")
    
        #plot the best rank-s approximation of image:
        plt.subplot(1,2,2)
        plt.imshow(approx_image, cmap = "gray")
        plt.title("Approximated Image")
        
        plt.show()
        
    else:  #dont have to do anything special for the grayscale image
        approx_image = svd_approx(image, s)[0]   #have [0] because only want A_s from problem 3, so only want 1st thing problem 3 returns
                                                 #approximated image
        size_approx_image = svd_approx(image, s)[1]  #get size of the approximated image when its gray
   
        #plot OG image and approx image in different subplots:
        
        image_size_diff = og_image_size - size_approx_image   #get the size difference between the 2 images
        plt.suptitle("Entries required to store OG image vs approximated image: " + str(image_size_diff))
    
        #plot OG image 1st:
        plt.subplot(1,2,1)
        plt.imshow(image, cmap = "gray")
        plt.title("Original Image")
    
        #plot the best rank-s approximation of image:
        plt.subplot(1,2,2)
        plt.imshow(approx_image, cmap = "gray")
        plt.title("Approximated Image")
    
        plt.show()
    
   
