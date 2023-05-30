import numpy as np
import matplotlib.pyplot as plt

from HEGData.LDAFits import fmap, zetamap

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3))

# Fit on exchange energy

zeta = np.sqrt(np.linspace(0., 1., 501))
fx_zeta = ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2
fb_fit = fmap(zeta)
fx_fit = (2/fb_fit)**(1/3)

ax1.plot(zeta**2, (fx_fit/fx_zeta - 1)*100, "-k")
ax1.text(0.3,0.1,
         "Using $\\bar{f}=\\hat{f}_{\\rm x-map}(\\zeta)$",
         transform=ax1.transAxes,
         fontsize=14,
         )
ax1.set_xlabel("$\\zeta^2$ [unitless]", fontsize=14)
ax1.set_xticks([0,0.5,1])
ax1.axis([0,1,-0.3,0.3])


fb = np.linspace(2., 1., 501)
fx_fb = (2/fb)**(1/3)
zeta = zetamap(fb)
fx_fit = ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2

ax2.plot(2-fb, (fx_fit/fx_fb - 1)*100, "-k")
ax2.text(0.3,0.1,
         "Using $\\zeta=\\hat{f}_{\\rm x-map}^{-1}(\\bar{f})$",
         transform=ax2.transAxes,
         fontsize=14,
         )
ax2.set_xlabel("$\\bar{f}$ [unitless]", fontsize=14)
ax2.set_xticks([0,0.5,1], [2,1.5,1])
ax2.axis([0,1,-0.3,0.3])

fig.supylabel("Error on $\\epsilon_{\\rm x}$ [%]", fontsize=14)

fig.tight_layout(pad=0.3)

plt.show()
