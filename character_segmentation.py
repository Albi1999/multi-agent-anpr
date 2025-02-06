import cv2 
import numpy as np 
import re 
import os

class CharacterSegmentation:
    def __init__(self, image):
        self.image = image




    def show_image(self):
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def order_points(self, pts):
        """Order points in clockwise order starting from top-left"""
        # Initialize ordered coordinates
        rect = np.zeros((4, 2), dtype=np.float32)

        # First, we need to understand which point is which
        # For this, let us first order the points by their x coordinate 
        pts_sorted_x = pts[pts[:,0].argsort()]
        # We know that points 0 and 1 are the left points, and 2 and 3 are the right points
        pair_left = pts_sorted_x[:2]
        pair_right = pts_sorted_x[2:]
        # Now we order these pairs by their y coordinate
        # The top left point will have the highest y coordinate of pair_left
        # The bottom left point will have the lowest y coordinate of pair_left
        # same idea for right pair
        pair_left_sorted_y = pair_left[pair_left[:,1].argsort()]
        pair_right_sorted_y = pair_right[pair_right[:,1].argsort()]
        rect[1] = pair_left_sorted_y[1]  # top-left
        rect[3] = pair_right_sorted_y[1] # top-right
        rect[0] = pair_left_sorted_y[0] # bottom-left
        rect[2] = pair_right_sorted_y[0] # bottom-right
    
        return rect


    def select_corners(self):
        padding = 20
        img_copy = self.image.copy()
        img_display = cv2.copyMakeBorder(img_copy, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        src = []
        selected_point = None

        def draw_lines():
            display = img_display.copy()
            for point in src:
                cv2.circle(display, tuple(point), 3, (0, 255, 0), -1)
            if len(src) > 1:
                for i in range(len(src)):
                    cv2.line(display, tuple(src[i]), tuple(src[(i+1)%len(src)]), (0,0,255), 2)
            cv2.imshow('Select Corners', display)

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_point
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(src) < 4:
                    src.append([x, y])
                else:
                    for i, point in enumerate(src):
                        if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                            selected_point = i
                            break
            elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
                src[selected_point] = [x, y]
            elif event == cv2.EVENT_LBUTTONUP:
                selected_point = None
            draw_lines()

        cv2.namedWindow('Select Corners')
        cv2.setMouseCallback('Select Corners', mouse_callback)
        print("Click 4 corner points. Press 'r' to reset, 'Enter' when done")

        while True:
            draw_lines()
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                src.clear()
            elif key == ord('f') and len(src) == 4:
                cv2.destroyAllWindows()
                src = np.array(src) - padding
                return np.float32(src)
            elif key == 27:
                cv2.destroyAllWindows()
                break


    def adaptive_thresholding(self, block_size=25, constant=1):
        current_thresh = None  # Store current threshold image


        def on_change(_):
            nonlocal current_thresh
            block = cv2.getTrackbarPos('Block Size', 'Adjust Parameters') 
            const = cv2.getTrackbarPos('Constant', 'Adjust Parameters')
            block = block * 2 + 1 # always keep block size an uneven number (else error)
            
  
            img_show = self.image.copy()
                
            lab = cv2.cvtColor(img_show, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            thresh = cv2.adaptiveThreshold(
                l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block, const
            )
            current_thresh = thresh  # Update current threshold
            cv2.imshow('Output', thresh)

        cv2.namedWindow('Adjust Parameters')
        cv2.namedWindow('Output')
        print("Press 'f' when finished or 'ESC' to cancel")
        
        cv2.createTrackbar('Block Size', 'Adjust Parameters', (block_size-1)//2, 100, on_change)
        cv2.createTrackbar('Constant', 'Adjust Parameters', constant, 50, on_change)

        on_change(0)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                cv2.destroyAllWindows()
                return current_thresh
                # changed : normally just return current_thresh, but need to generate some of the images
                # to check what we need for the synthetic dataset 
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                break


   
    def scale_and_resize(self, target_height=550):
        # GERMAN LICENSE PLATES : 520mmx110mm 
        """Scale and resize the license plate to the given dimensions."""

        # Standard height for license plates while maintaining aspect ratio
        aspect_ratio = self.image.shape[1] / self.image.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Resize while maintaining aspect ratio
        self.image = cv2.resize(self.image, (target_width, target_height))
        return self.image



    def perspective_correction(self):


        src = self.select_corners()


        # Get the source points (i.e. the 4 corner points)
       # src = np.squeeze(license_cont).astype(np.float32)

        height = self.image.shape[0] 
        width = self.image.shape[1]
        # Destination points (for flat parallel)
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        license_cont = self.order_points(src)
     #   dst = self.order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(license_cont, dst)

        # Warp the image
        img_shape = (width, height)
        self.image = cv2.warpPerspective(self.image, M, img_shape, flags=cv2.INTER_LINEAR)

        print(self.image.shape)

     #   return self.image


    def character_segmentation(self):

        padding = 30
        img_copy = self.image.copy()
        # Add some padding
        img_copy = cv2.copyMakeBorder(img_copy, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        src = []


     #   img_copy = self.image.copy()
        points = []
        counter = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, img_copy
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
                
                if len(points) == 4:
                    points_arr = np.array(points)
                    x_sorted = points_arr[np.argsort(points_arr[:, 0])]
                    left = x_sorted[:2]
                    right = x_sorted[2:]
                    left = left[np.argsort(left[:, 1])]
                    right = right[np.argsort(right[:, 1])]
                    
                    sorted_points = np.array([left[0], right[0], right[1], left[1]], dtype=np.int32)
                    cv2.polylines(img_copy, [sorted_points], True, (255, 0, 0), 2)
        
        cv2.namedWindow('Select Characters')
        cv2.setMouseCallback('Select Characters', mouse_callback)
        
        while True:
            cv2.imshow('Select Characters', img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if len(points) == 4 and key == ord('f'):
                points_arr = np.array(points)
                x = min(points_arr[:, 0])
                y = min(points_arr[:, 1])
                w = max(points_arr[:, 0]) - x
                h = max(points_arr[:, 1]) - y
                
                roi = self.image[y:y+h, x:x+w]
              #  scale = min(20.0/w, 20.0/h)
              #  new_w = int(w * scale)
              #  new_h = int(h * scale)
              #  char_resized = cv2.resize(roi, (new_w, new_h))
                
              #  mnist_size = np.zeros((28, 28), dtype=np.uint8)
              #  x_offset = (28 - new_w) // 2
              ##  y_offset = (28 - new_h) // 2
              #  mnist_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized

             #   mnist_size = self.preprocessing(mnist_size)
                
                os.makedirs(f"VCS_Project/results/license_plates/segmented_plate.{517}", exist_ok=True)
                cv2.imwrite(f"VCS_Project/results/license_plates/segmented_plate.{517}/character_{counter}.png", roi)
                
                counter += 1
                points = []
                # Reset for next iteration
                img_copy = self.image.copy()
                img_copy = cv2.copyMakeBorder(img_copy, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                
            elif key == 27:  # ESC
                break
                
        cv2.destroyAllWindows()
        return None




if __name__ == '__main__':
    image_path = 'VCS_Project/results/noisy_license_plates/license_plate.800.png'
    image = cv2.imread(image_path)
    curr_img = CharacterSegmentation(image)
 #   curr_img.show_image()
    curr_img.scale_and_resize()
    curr_img.perspective_correction()
    curr_img.character_segmentation()
 #   curr_img.show_image()
   # curr_img.skeleton()
  #  curr_img.adaptive_thresholding()
 #   curr_img.extract_skeleton()
  #  curr_img.show_image()

