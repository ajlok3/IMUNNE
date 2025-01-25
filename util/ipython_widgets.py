import torchvision
import ipywidgets as widgets
from IPython.display import display
import torch
from torchvision import transforms
import io

# Define the image transform to convert Torch tensors to PIL images
transform = transforms.ToPILImage()

def imagescn(img,width=900):
    
    # Create a slider widget to control the image index
    slider = widgets.IntSlider(min=0, max=len(img)-1, value=0, description='Image Index')

    # Create an image widget to display the selected image
    image_widget = widgets.Image(format='png', width=width)
    
    def update_image(index):
        # Convert the Torch tensor to a PIL image
        im = torch.concat([img[index].abs().cpu() for i in range(3)], dim=0)
        im = im/im.max()*1.5
        im = torch.clip(im,max=1)
        print(im.shape)
        pil_image = transform(im.permute(0,2,1))

        img_byte_arr = io.BytesIO()
        # Convert the PIL image to bytes
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        #print(image_bytes)
        # Update the image widget with the new image
        image_widget.value = image_bytes
    
    # Register the update_image function as a callback for slider value changes
    widgets.interact(update_image, index=slider)

    # Display the slider and image widgets

    display(image_widget)
