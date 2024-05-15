import cv2
import numpy as np

def calculate_contour_similarity(image1, image2):
    # Check if images are None
    if image1 is None or image2 is None:
        print("Error: One or both images are None.")
        return 0.0

    # Convert images to grayscale if necessary
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1

    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2

    # Find contours in the images
    contours1, _ = cv2.findContours(gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate contour similarity score
    contour_similarity = len(contours1) / len(contours2) if len(contours2) != 0 else 0
    
    return contour_similarity

def calculate_histogram_similarity(image1, image2):
    # Check if images are None
    if image1 is None or image2 is None:
        print("Error: One or both images are None.")
        return 0.0

    # Check if images have valid shapes
    if image1.shape != image2.shape:
        print("Error: Image shapes do not match.")
        return 0.0

    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Calculate histogram comparison scores
    # intersection_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    correlation_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # chi_square_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    # Calculate average similarity score
    # average_score = (intersection_score + correlation_score + chi_square_score) / 3
    
    return correlation_score

def calculate_image_similarity(image1, image2, contour_weight=0.5, histogram_weight=0.5):
    contour_similarity = calculate_contour_similarity(image1, image2)
    histogram_similarity = calculate_histogram_similarity(image1, image2)
    
    # Normalize weights
    total_weight = contour_weight + histogram_weight
    contour_weight /= total_weight
    histogram_weight /= total_weight
    
    # Combine scores using weighted average
    combined_score = (contour_similarity * contour_weight) + (histogram_similarity * histogram_weight)
    
    return combined_score


image1 = cv2.imread('/home/s3977773@RMIT.EDU.VN/Documents/GitHub/Assignment-2/Data/Furniture_Data/tables/Craftsman/21101craftsman-console-tables.jpg')
image2 = cv2.imread('/home/s3977773@RMIT.EDU.VN/Documents/GitHub/Assignment-2/Data/Furniture_Data/tables/Craftsman/122.jpg')


# contour_similarity = calculate_contour_similarity(image1, image2)
# print("Contour similarity score:", contour_similarity)

# histogram_similarity = calculate_histogram_similarity(image1, image2)
# print("Histogram similarity score:", histogram_similarity)

similarity_score = calculate_image_similarity(image1, image2)
print("Combined similarity score:", similarity_score)
