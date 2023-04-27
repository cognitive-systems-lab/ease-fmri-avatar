import qrcode
import matplotlib.pyplot as plt
import numpy as np
input_data = f"{1}"#Creating an instance of qrcode
qr = qrcode.QRCode(
        version=1,
        box_size=2,
        border=2)
qr.add_data(input_data)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')
plt.imshow(img)
print(np.asarray(img).shape)
plt.show()
