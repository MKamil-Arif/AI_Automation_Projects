# DIP Studio: All-in-One Digital Image Processing Project

24 DIP topics ek hi app mein. Har topic ki apni file hai aur sab ek common OOP base class se bante hain.
Har topic **asli (real) datasets** par test hua hai. Poori tables [EVALUATION.md](EVALUATION.md) mein hain.

## Run kaise karein

```bash
pip install -r requirements.txt          # torch, easyocr aur ultralytics ke notes requirements.txt mein hain
python samples/generate_samples.py       # test images banata hai (sirf pehli dafa)
streamlit run app.py                     # browser mein app khul jayegi
```

Kisi topic ko UI ke baghair terminal se bhi chala sakte hain. Output aur har step ki image `outputs/topic_XX/` mein save hoti hai:

```bash
python -m topics.t23_blood_cell_counter samples/blood_smear_synthetic.png method=Watershed
python -m topics.t19_invisible_watermarking my_photo.jpg "mode=Robustness Test"
python -m topics.t12_cbir_search_engine samples/cbir_query_sea.jpg "mode=Evaluate database"
python -m topics.t06_fingerprint_matching samples/fingerprint_A1.png "mode=Match two fingerprints" second=samples/fingerprint_A2.png
python -m topics.t05_number_plate_recognition samples/car_plate_LEB_1234.png
python -m topics.t09_image_captioning samples/face_astronaut.png "backend=Gemini (cloud)" urdu=true
python -m topics.t11_emotion_recognition --webcam          # real-time webcam window (ESC = band)
```

Parameters ki list dekhne ke liye file ko bina image ke chalayein, jaise `python -m topics.t23_blood_cell_counter`.

### Real data par evaluation

```bash
python evaluation/fetch_datasets.py      # saare real datasets datasets/real/ mein (sirf zaroori hissa)
python evaluation/evaluate_all.py        # har topic ki accuracy -> evaluation/results.json
python evaluation/evaluate_all.py fer sr # sirf kuch topics
python evaluation/make_report.py         # results.json -> EVALUATION.md
python evaluation/robustness_test.py     # har topic ko ajeeb inputs (khali, gray, 4000x3000 ...)
python evaluation/train_detectors.py potholes   # YOLOv8n ko dobara train karna (plates bhi)
```

Models dobara train karne ke commands: `python -m topics.t10_super_resolution train`,
`python -m topics.t11_emotion_recognition calibrate datasets/real/fer2013_val`,
`python -m topics.t24_crop_weed_detection train datasets/real/cwfid`,
`python -m topics.t22_fruit_ripeness train datasets/real/fruit_train`,
`python -m topics.t08_mask_helmet_detection train-mask datasets/real/face_mask`.

## API keys (.env)

Project folder mein `.env` file ho:

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

- **Gemini** (`gemini-3.5-flash`): images samajhta hai. Caption (9), zero-shot emotion (11), mask/helmet (8), fish (2) aur plate OCR (5) mein "cloud" option.
- **Groq** (`gpt-oss`): sirf text. Caption ka Urdu tarjuma (9) aur bimari ke ilaaj ke mashware (20).
- Keys sirf [core/cloud.py](core/cloud.py) parhta hai aur sirf request headers mein bhejta hai. Na kahin print hoti hain, na app mein dikhti hain. `.env` file `.gitignore` mein hai.
- Har topic ka offline method bhi hai, to internet ke baghair bhi app chalti hai (viva ke liye acha hai).
- Note: is free key par Gemini image *generation* ka quota zero hai. Image samajhna kaam karta hai.

## Real data par accuracy (summary)

Har number aisi images par hai jo training mein **istemal nahi** hui. Thresholds bhi alag data par chune gaye.

| # | Topic | Real dataset (test) | Result |
|---|---|---|---|
| 1 | Tree Counting | AgML olive orchard drone images (12) | count accuracy **86.3%** |
| 2 | Fish Species | phylo-fish, 8 species (25% test) | **97.4%** |
| 3 | Flower Species | TF-Flowers, 5 classes, 918 test photos | deep **94.8%**, classical 60.9% |
| 4 | Face Recognition | LFW 1000 pairs | **98.9%** (ROC AUC 0.998) |
| 5 | Number Plate | 200 car photos (plate dhoondhna), 437 photos with text (parhna) | plate mili **97%** (classical 27%); parhna: character accuracy **66%**, poora number 10.3% |
| 6 | Fingerprint | FVC2002 DB1_B (threshold DB2_B se) | accuracy **77.2%**, false match 0%, EER 11.1% |
| 8 | Mask / Helmet | Kaggle face-mask (200), hard-hat crops (184) | mask **100%**, helmet **98.4%** |
| 9 | Image Caption | Flickr8k, 100 photos × 5 human captions | BLEU-1 **0.73**, BLEU-4 **0.23** |
| 10 | Super Resolution | Set5 / Set14 / BSD100, x2 / x3 / x4 | bicubic se **+0.53 se +1.04 dB** (Set5 x2: 32.64 → 33.55 dB) |
| 11 | Emotion | FER-2013 test, 700 faces | **61.9%** (raw FER+ 49.1%) |
| 12 | CBIR | Corel-1K, 1000 queries | P@10 **98.1%**, mAP **93.9%** |
| 13 | Potholes | 25 road photos (YOLO ne kabhi nahi dekhi) | YOLO F1 **61.3%** (classical 1.8%) |
| 14 | Waste | TrashNet, 6 classes (25% test) | **92.9%** |
| 16 | Old Photo Restoration | 20 photos + simulated damage | SSIM 0.544 → **0.619** |
| 17 | Background Remover | DUTS-TE, 100 photos | U2-Net IoU **0.76**, F 0.85 (GrabCut 0.38–0.44) |
| 18 | Copy-Move Forgery | 30 Flickr photos, copy-move + JPEG 90 | **98.3%** (96.7% pakra, 0 false alarm) |
| 19 | Watermark | 20 Flickr photos, 8 attacks | JPEG-50 ke baad **80%** bacha, PSNR 36.6 dB |
| 20 | Leaf Disease | PlantVillage, 114 photos, 38 classes | **95.6%** |
| 21 | Leaf Species | medicinal leaves, 6 species (25% test) | **100%** (classical 99.4%) |
| 22 | Fruit Ripeness | banana / mango, 200 test photos | SVM **93.5%** (colour rules 36%) |
| 23 | Blood Cells | BCCD, 364 smears | WBC exact **95.3%**, RBC MAE 5.9 |
| 24 | Crop / Weed | CWFID test (20) | vegetation IoU **0.85**, crop/weed pixel acc **78.5%** |

Topic 7 (currency) aur 15 (photo editor) ka koi public labelled dataset nahi; ye synthetic/apni photos par test hue.

**Is round mein theek hone wale bugs (report ke liye acha point):**
- Topics 3 aur 21 mein "Classical (colour + shape)" option asal mein classical + CNN dono chala raha tha, kyunke naam mein "+" tha. Model bhi ghalat file mein save hota tha. Ab har option apni file mein save hota hai.
- SR (10) ke liye benchmark images OpenCV ke non-antialiased bicubic se chhoti ho rahi theen, jabke training INTER_AREA par hui thi. Is mismatch se SRCNN bicubic se *kharab* lag raha tha. Ab dono jagah MATLAB jaisa antialiased bicubic hai.
- Set5 ki "woman" image BSDS500 training photos mein bhi thi. Usay training se nikala.
- Fingerprint threshold (30%) synthetic prints par set tha, real prints par 75% genuine reject ho rahe the. Ab 9% hai (DB2 par koi false match nahi).
- CWFID split file mein image 28 train aur test dono mein hai. Test se nikali.
- Windows terminal par CLI "∞" print karte hue crash hota tha.
- Tree counting ka "bahut bare blob hatao" filter olive orchard ke patton ko bhi mita raha tha.
- Copy-move threshold 60 par 10% asli photos ko forged keh raha tha. Ab 120 hai (tuning alag photos par).

## Downloaded models aur datasets

| Cheez | Size | Kahan se | Folder | Topic |
|---|---|---|---|---|
| YuNet face detector | 0.2 MB | github.com/opencv/opencv_zoo | models/ | 4, 8, 11 |
| SFace face recognizer | 37 MB | github.com/opencv/opencv_zoo | models/ | 4 |
| FER+ emotion CNN | 34 MB | github.com/onnx/models | models/ | 11 |
| BLIP-base captioning | ~1 GB | huggingface.co/Salesforce | models/hf/ | 9 |
| PlantVillage MobileNetV2 (ONNX) | 9 MB | huggingface.co/onnx-community | models/plant_disease/ | 20 |
| ConvNeXt-Tiny + MobileNetV3 ImageNet weights | 130 MB | download.pytorch.org | models/torch/ | 2, 3, 8, 12, 14, 21 |
| EasyOCR detector + reader | 94 MB | EasyOCR ka GitHub release | models/easyocr/ | 5 |
| YOLOv8n (base, fine-tune ke liye) | 6 MB | ultralytics | – | 5, 13 |
| U2-Net-p salient object (ONNX) | 4.4 MB | github.com/danielgatis/rembg releases | models/ | 17 |

Hamare khud train kiye hue models (`models/`):

| File | Topic | Kis data par |
|---|---|---|
| `pothole_yolov8n.pt`, `plate_yolov8n.pt` | 13, 5 | keremberke pothole (65 photos), license-plate (400 photos) |
| `srcnn_x234.pth`, `srcnn_gain.json` | 10 | BSDS500 ki 389 photos (gain: 10 validation photos) |
| `fer_calibration.json` | 11 | FER-2013 ki 1000 *train* faces |
| `mask_svm`, `helmet_svm` | 8 | face-mask 800 photos, hard-hat head crops |
| `flower_svm_*`, `leaf_svm_*` | 3, 21 | TF-Flowers 3670, medicinal leaves 1380 (har feature set ki apni file) |
| `waste_svm`, `fish_svm` | 14, 2 | TrashNet, phylo-fish |
| `fruit_ripeness_svm` | 22 | 600 banana / mango photos |
| `cropweed_svm` | 24 | CWFID ki 40 train images |

Real datasets `evaluation/fetch_datasets.py` download karta hai: BCCD, AgML olive trees, TF-Flowers, medicinal leaves, FER-2013,
Corel-1K, Kaggle face-mask, hard-hat, license plates (+ text), potholes, PlantVillage, Flickr8k, Set5/14/BSD100, BSDS500, LFW,
FVC2002, DUTS-TE, CWFID, fruit ripeness. Sirf zaroori hissa download hota hai, gigabytes nahi.

## Folder structure

```
All in one/
├── app.py                  Streamlit UI (saray topics, ek app)
├── .env                    API keys (share na karein)
├── core/
│   ├── base.py             DIPModule (abstract class), Param, Step, ModuleResult
│   ├── registry.py         @register decorator + 24 topics ki list
│   ├── utils.py            read/write, resize, histogram, PSNR, SSIM, granulometry ...
│   ├── recognition.py      FeatureClassifier (SVM) + RecognitionModule (topics 2, 3, 14, 21)
│   ├── deep.py             CNNEmbedder: pretrained ConvNeXt-Tiny features (transfer learning)
│   ├── yolo.py             YoloDetector: hamare fine-tuned YOLOv8n (topics 5, 13)
│   ├── faces.py            YuNet face detector (topics 4, 8, 11)
│   ├── cloud.py            Gemini + Groq clients (.env keys)
│   └── cli.py              terminal se chalane ka helper
├── topics/                 HAR TOPIC KI APNI FILE: t01_... se t24_... tak
├── evaluation/             fetch_datasets, evaluate_all, train_detectors, make_report, robustness_test
├── samples/                test images + generate_samples.py (+ kuch real photos)
├── datasets/               downloaded datasets (git mein nahi)
├── models/                 downloaded + hamare trained models
├── faces_db/               topic 4: enrolled logon ki photos (khud banta hai)
├── attendance/             topic 4: attendance_log.csv + attendance.xlsx
└── outputs/                terminal runs ke results, CBIR index cache
```

## OOP concepts (viva ke liye)

| Concept | Kahan use hua |
|---|---|
| **Abstraction** | `DIPModule` abstract class hai. Har topic ko `process()` likhna zaroori hai, warna object nahi banega. |
| **Inheritance** | `class BloodCellCounter(DIPModule)`: har topic base class se inherit karta hai. |
| **Multi-level inheritance** | `DIPModule → RecognitionModule → FlowerRecognizer / LeafRecognizer / WasteSegregation / FishRecognizer`. Training, saving aur predict ek hi jagah likha hai. Topic sirf features aur `explain()` deta hai. Flower aur Leaf mein classical, deep (CNN) ya dono features chun sakte hain. |
| **Method overriding + super()** | `WasteSegregation.process()` aur `FishRecognizer.process()` pehle `super().process()` chalate hain, phir apni cheez jorte hain (bin advice, underwater correction, Gemini). |
| **Encapsulation** | Pipeline ke steps private list `__steps` mein hain. Bahar se sirf `_add_step()` aur read-only `steps` property milti hai. API keys sirf `core/cloud.py` ke andar. |
| **Polymorphism** | `app.py` kisi bhi topic par `module.run(image)` call karti hai, bina jaane ke wo konsa topic hai. |
| **Template Method pattern** | `run()` order fix karta hai (validate → reset → process). Subclass sirf `process()` bharti hai. |
| **Composition** | `FruitRipenessDetector` ke andar `ColorHistogramClassifier`. `FaceAttendance` ke andar `FaceGallery`. `RecognitionModule` ke andar `FeatureClassifier`. |
| **Strategy pattern** | Topic 12: `FeatureExtractor` abstract class, aur 6 features (deep CNN bhi) uski alag classes. |
| **Singleton / shared instance** | `FaceDetector.shared()`, `CNNEmbedder.shared()`, `YoloDetector.get()`: bara model sirf ek dafa load hota hai. |
| **Class variable / caching** | `ImageIndex._cache` (+ disk cache), `FaceGallery._cache`, BLIP / FER+ / SFace / U2-Net models class variables mein. |
| **Decorator / Registry** | `@register` lagate hi topic app mein khud aa jata hai. |
| **Dataclasses** | `Param`, `Step`, `ModuleResult`, `Face`, `FruitProfile`, `CropRows`, `FingerprintAnalysis`, `NoteTemplate` |

## Topics status: 24 / 24 ✅

| # | Topic | Method (short) | Synthetic test | Real data |
|---|---|---|---|---|
| 1 | Tree Counting | ExG + Otsu (chhota hissa = darakht) + granulometry + watershed / LoG | 36/36 | 86.3% count acc |
| 2 | Fish Species | underwater correction + ConvNeXt + SVM, ya Gemini | 8/8 queries | 97.4% |
| 3 | Flower Species | ConvNeXt features (default) ya colour + petal FFT + SVM | 4/4 | 94.8% |
| 4 | Face Attendance | YuNet + SFace + cosine, CSV/Excel | sahi pehchaan | LFW 98.9% |
| 5 | Number Plate (ANPR) | YOLOv8n (trained) ya black-hat + Sobel, EasyOCR / Gemini | 3/3 plates | plate 97%, characters 66% |
| 6 | Fingerprint | Gabor + crossing number + Hough alignment | genuine ≥ 0.57, impostor ≤ 0.09 | 77.2%, 0 false match |
| 7 | Fake Currency | ORB homography + 4 checks | 20/20 photos | – |
| 8 | Mask & Helmet | ConvNeXt + SVM (default), skin rule, ya Gemini | 4/4 | 100% / 98.4% |
| 9 | Image Caption | BLIP offline / Gemini + Urdu (Groq) + k-means colours | maqool captions | BLEU-4 0.23 |
| 10 | Super Resolution | SRCNN (residual) khud train kiya, per-scale detail gain | – | +0.5 – 1.0 dB |
| 11 | Emotion | YuNet + FER+ (mirror TTA + calibration), ya Gemini | smiling → happy | 61.9% |
| 12 | CBIR | 6 features (deep default) × 4 distances | P@8 98.8% | P@10 98.1% |
| 13 | Pothole & Crack | YOLOv8n (trained) ya darkness; Sato ridge for cracks | 4/4 potholes | F1 61.3% |
| 14 | Waste Segregation | ConvNeXt + SVM (TrashNet) + bin advice | 5/6 queries | 92.9% |
| 15 | Photo Editor | WB, stretch, gamma, CLAHE, filters | – | – |
| 16 | Old Photo Restoration | top-hat + inpainting + Welsh colorization | SSIM 0.22 → 0.68 | SSIM 0.54 → 0.62 |
| 17 | Background Remover | U2-Net (default) / GrabCut / chroma + guided filter | IoU 0.997 | IoU 0.76 |
| 18 | Copy-Move Forgery | DCT block matching / SIFT | JPEG-75 ke baad bhi | 98.3% |
| 19 | Watermarking | DWT + QIM + key (strength 36) | 8/8 attacks | JPEG-50: 80% |
| 20 | Leaf Disease | PlantVillage CNN + lesion % + Groq advice | 8/8 samples | 95.6% |
| 21 | Leaf Species | shape + Fourier + roughness + ConvNeXt + SVM | 100% | 100% |
| 22 | Fruit Ripeness | colour-histogram SVM (default) / HSV rules | 12/12 (rules) | 93.5% |
| 23 | Blood Cells | HSV + granulometry + watershed / Hough | 70/70 RBC | WBC 95.3% |
| 24 | Crop / Weed | ExG + rows / plant size / trained plant SVM | rows sahi, ±2 plants | 78.5% pixel acc |

## Topics ki tafseel

- **1 Trees:**
  - Otsu greenness ko do hisson mein baant-ta hai. Darakht hamesha *chhota* hissa hote hain. Mitti par wo zyada hare hote hain, ghaas par (silvery olive) kam hare.
  - Crown ka size granulometry (pattern spectrum) se khud nikalta hai, slider 0 = auto.
  - Watershed ka minimum area (0.75) olive images par chuna gaya. Ye 12 images par chuna aur unhi par report hua, is liye number thora optimistic ho sakta hai.
- **2 Fish:**
  - Underwater colour correction (classical): red channel compensation (Ancuti 2018), gray-world white balance, aur CLAHE.
  - Species ConvNeXt-Tiny ke 768 features + SVM, ya Gemini zero-shot.
- **3 Flowers / 21 Leaves:**
  - Teen feature sets hain: classical, deep aur dono. Har ek ka model alag save hota hai.
  - Flowers par classical sirf 60.9% aur deep 94.8% hai, is liye default deep hai. Leaves par classical bhi 99.4% hai.
  - Default models real photos par trained hain (TF-Flowers, 6 medicinal leaves). Apni species ke liye folders bana kar "Train from folder" karein.
- **4 Attendance:** SFace threshold 0.363 (SFace ki published value). LFW par 98.9%, aur best threshold 0.34 hai, yani published value theek hai.
- **5 ANPR:**
  - Plate YOLOv8n se dhoondi jati hai (400 car photos par fine-tune). Purana classical tareeqa (black-hat + Sobel + closing) option ke tor par hai.
  - EasyOCR do-line plates ko line ba line parhta hai. Pakistani format (ABC 1234) ki clean-up option hai.
  - YOLO detector held-out photos par mAP50 0.992 deta hai. Plate dhoondhna ab masla nahi, asal kamzori OCR hai.
  - OCR se pehle plate ko 128 px height tak bara kiya jata hai, phir gray + CLAHE. Ye setting dataset ki *validation* split par chuni gayi. Test par character accuracy 64% → 66% aur poora number 6% → 10.3% hua.
  - Poora number kam is liye sahi aata hai:
    - zyada tar plates Vietnamese/Indian format ki hain
    - EasyOCR "5" ko aksar "6" ya "B" parhta hai
    - kuch labels khud ghalat hain (jaise `MH 20.80 20`, jabke plate `MH 20 BQ 20` hai)
  - Gemini (cloud) jab jawab de to behtar hai: 20 mai se 12 bilkul sahi (60%). Lekin free quota par 60 mai se 40 requests fail hui.
- **6 Fingerprint:** real FVC prints synthetic se bohat kam score karte hain (same finger ~0.21, alag ~0.01). Threshold 9% DB2 par chuna gaya (koi false match nahi) aur DB1 par test hua.
- **8 Mask & Helmet:** default ConvNeXt + SVM. Classical skin rule real photos par 77% (jab chehra mile) aur Gemini cloud option bhi hai.
- **9 Caption:**
  - BLIP pehli dafa load hone mein ~40 s leta hai, phir har caption ~5 s (CPU par).
  - Viva point: BLIP ne astronaut (aurat) ko "a man" kaha. Ye training data ka bias hai.
- **10 SRCNN:**
  - Network: 9×9 → 1×1 → 5×5 conv, Y channel par, residual learning ke saath (VDSR ka idea).
  - Low-res images MATLAB jaisi antialiased bicubic se banti hain, jo SR benchmarks ka standard tareeqa hai.
  - Training BSDS500 ki 389 photos par hui. Set5 / Set14 / BSD100 training mein nahi hain.
  - Ek hi network x2, x3 aur x4 teeno karta hai. x2 par wo zarurat se zyada detail dalta tha (Set5 x2 par bicubic se -0.16 dB). Is liye har scale ki "detail gain" alag hai (x2: 0.7, x3: 1.1, x4: 1.5). Ye gains 10 validation photos par chuni gayi, test benchmarks par nahi (`python -m topics.t10_super_resolution calibrate`).
  - Nateeja: purana model bicubic se sirf +0.17 se +0.49 dB behtar tha, naya **+0.53 se +1.04 dB**.
- **11 Emotion:**
  - FER+ asli angry / sad / fearful chehron ko aksar "neutral" kehta hai.
  - Face aur uski mirror image dono score hoti hain. Phir FER-2013 ki 1000 *train* faces par fit hui logistic regression 7 emotions deti hai: 49% → 62%.
  - "Calibrated" checkbox band karke raw FER+ (8 emotions) dekh sakte hain. Fear ab bhi mushkil hai (25%).
- **12 CBIR:** deep CNN features Corel-1K par P@10 98.1%, classical mein best colour + texture 77%. Feature index disk par cache hota hai (`outputs/cbir_index`).
- **13 Pothole:** YOLOv8n 65 photos par trained. Classical darkness method asli sarkon par kaam nahi karta (F1 1.8%): saaye, tar aur patches bhi dark hote hain. Ye report ka acha point hai.
- **14 Waste:** TrashNet mein "organic" class nahi hai. `datasets/trashnet/organic/` mein apni photos daal kar dobara train karein.
- **16 Old Photo:** pehle contrast stretch, phir top-hat aur shape filter (lambi-patli lakeerein ya chhote dhabbe), phir inpainting.
- **17 Background:** default U2-Net-p (4.4 MB CNN, offline). GrabCut rectangle maanta hai ke subject frame bhar raha hai, is liye DUTS ki chhoti objects par kamzor hai. Guided filter kinare ki patti mein lagta hai.
- **18 Copy-Move:** smooth area (aasman, paani) wali copy dono methods miss karte hain. Threshold 120: asli photos max 74 tak gayin.
- **19 Watermarking:**
  - YCrCb → Haar DWT → QIM. Copies key se scramble aur XOR hoti hain, extraction majority vote se hoti hai (blind).
  - Default strength 36 hai: PSNR ~37 dB (nazar nahi aata) aur JPEG-50 ke baad 80% bachta hai. Strength 20 par PSNR 42 dB tha lekin JPEG-50 ke baad 0%.
  - Blur ke baad 60% aur resize ke baad 40% bachta hai.
- **20 Leaf Disease:** evaluation mein folders case-sensitive naam se sort hone chahiye. Windows par sorting case ignore karti thi, jis se pehle accuracy ghalti se 52% aayi thi.
- **22 Fruit:** real photos par colour rules sirf 36% hain: saaye "dark/rot" ban jate hain aur kachay olive-green aam peelay lagte hain. SVM 93.5% hai. Dataset mein rotten fruit nahi, is liye "Rotten" sirf colour rules kehte hain.
- **23 Blood Cells:** BCCD par WBC 95% exact. RBC labels adhoore hain (bohat cells label nahi), is liye RBC error pessimistic hai. Platelets mushkil hain (count accuracy 25%). Train/test split par tuning se bhi behtar nahi hua.
- **24 Crop/Weed:**
  - Drone images mein rows wala rule hai. Close-up photos (CWFID, sirf ek row) ke liye 11 shape/colour features + SVM wala "Trained" rule hai.
  - ExG - ExR ka fixed threshold 0 CWFID ke colour cast mein fail hota tha (IoU 0.24). Ab har photo ke liye Otsu threshold hai (IoU 0.85).

> `samples/` ki zyada tar images synthetic (computer se bani) hain. `face_astronaut` / `face_cameraman` scikit-image ki public-domain images hain. `flower_photo_*`, `leaf_photo_*`, `fruit_photo_*` real datasets se hain (`fetch_datasets.py` copy karta hai).

## Naya topic kaise add karein

`topics/t25_my_topic.py` naam ki file banayein:

```python
from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


@register
class MyTopic(DIPModule):
    topic_no = 25
    name = "My Topic"
    description = "..."

    def parameters(self):
        return [Param("threshold", "Threshold", "slider", 100, 0, 255, 1)]

    def process(self, image, *, threshold):
        gray = utils.ensure_gray(image)
        self._add_step("Gray", gray)
        ...
        return ModuleResult(output_image, info={"Result": value})
```

App mein topic khud aa jayega aur settings ke widgets bhi khud ban jayenge.
