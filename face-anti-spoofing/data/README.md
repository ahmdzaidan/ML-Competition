# Dataset Setup - FindIT! DAC 2026 Face Anti-Spoofing

The dataset for this competition consists of facial images categorized into six primary classes representing genuine faces and various presentation attack vectors:

### Class Definitions

| Class Name | Type | Description |
| :--- | :---: | :--- |
| `realperson` | **Bona Fide** | Genuine facial images of individuals without any manipulation or spoofing attempt. |
| `fake_printed` | **Spoof Attack** | Presentation attacks using printed photographs (print attack) to deceive the system. |
| `fake_screen` | **Spoof Attack** | Digital replay attacks displaying facial images via smartphone screens, tablets, or monitors. |
| `fake_mask` | **Spoof Attack** | Physical 3D or silicone face masks worn to mimic authorized individuals. |
| `fake_mannequin`| **Spoof Attack** | Artificial human face replicas, such as store mannequins and dolls. |
| `fake_unknown` | **Spoof Attack** | Miscellaneous and hybrid spoofing attack variations not covered by the defined categories. |

---

### Expected Directory Layout

Place or extract the competition dataset files in `data/` as follows:

```
data/
├── train/                    # Training images organized by class or labeled CSV
│   ├── realperson/
│   ├── fake_printed/
│   ├── fake_screen/
│   ├── fake_mask/
│   ├── fake_mannequin/
│   └── fake_unknown/
├── test/                     # Unlabeled test images (test_001.jpg, test_002.jpg, ...)
├── train.csv                 # (Optional/if provided) Image paths and labels
├── test.csv                  # Test image IDs to predict
└── sample_submission.csv     # Target format for submission
```

*Note: Raw images and archive files (`DAC2026.zip`) are excluded from git tracking via `.gitignore`.*
