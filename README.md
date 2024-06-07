# cofeHEG
 Code for the constant occupation factor ensemble (cofe) HEG. Main item of interest is:
 * HEGData/LDAFits.py

```python
import numpy as np
from HEGData.LDAFits import *

rs = np.logspace(-2,2,21)
fbar, zeta = 1.5, 0.66

epsx_cofe, epsc_cofe = LDA_cofe(rs, fbar)
epsxc_cofe = epsx_cofe + epsc_cofe
epsxc_rPW92 = LDA_rPW92(rs, zeta, Sum=True)
epsxc_PW92 = LDA_PW92(rs, zeta, Sum=True)
```

The paper uses:
* Plot_fq.py produces Figures 1 and 2
* NewLDAFits.py fits the correlation parametrization and produces Figure 3
* RPAcorrelation.py does the RPA calculations and produces Figures 4
* Plot_xiMap.py produces Figure 5 (after RPAcorrelation.py is run)

Additional data, including error data for molecules from TDLDA, MOM and eLDA, may be found in the Data folder.

