
Проект реализует мультимодальный пайплайн для распознавания скрытых эмоций по видео с использованием трех типов признаков:
- **context** — RGB-видеопризнаки;
- **face** — лицевые эмбеддинги;
- **skeleton** — признаки по keypoints/CSV.

## Основные этапы

1. **Подготовка данных**  
   Загружаются RGB- и skeleton-данные для Phase 1 и Phase 2, объединяются `train_label.csv` и `validation_label.csv`, формируется единая таблица метаданных с `video_id`, `label`, `split` и путями к файлам.

2. **Кэширование признаков**  
   Для каждого видео заранее извлекаются и сохраняются `.pt`-признаки по трем модальностям:
   - context: `r3d_18`, `mc3_18`, `r2plus1d_18`;
   - face: baseline, `FaceNet`, `EmotiEffLib`;
   - skeleton: `2s-AGCN`, `HRNet`, `PoseNet`.

3. **Формирование датасета**  
   Закэшированные признаки загружаются через кастомный `Dataset`, дополняются по длине и подаются в `DataLoader`.

4. **Обучение моделей на Phase 1**  
   Для каждой модальности обучаются отдельные трансформерные бинарные классификаторы с 3-fold stratified CV. Для борьбы с дисбалансом используются `WeightedRandomSampler`, `FocalLoss`, `EarlyStopping` и scheduler.

5. **OOF-оценка и отбор лучших моделей**  
   Для всех моделей собираются out-of-fold предсказания, считаются `Top-1` и `ROC-AUC`, после чего выбираются лучшие кандидаты внутри каждой модальности.

6. **Late fusion**  
   OOF-предсказания рангово нормализуются, затем перебираются комбинации `context + face + skeleton`, подбираются веса и порог бинаризации. Лучшая fusion-схема выбирается по `OOF Top-1`.

7. **Инференс на Phase 2**  
   Для тестовых данных загружаются лучшие модели, выполняется инференс по фолдам, затем предсказания усредняются и объединяются той же late-fusion схемой. На выходе формируется `submission.csv`.

## Google Drive paths / links

### Feature cache — Phase 1
[Google Drive: Phase 1 feature cache](https://drive.google.com/drive/folders/1wx77jdvzYPDFmXWWjKo76mGUg_Ghj3eR)

### Feature cache — Phase 2
[Google Drive: Phase 2 feature cache](https://drive.google.com/drive/folders/11bBvaHCe4l2YAy4m2Y9e6Lwa2lfvyUqY)

## Dataset URLs
```text
https://miga3.a3s.fi/imigue_rgb_phase1.zip
https://miga3.a3s.fi/imigue_rgb_phase2.zip
https://miga3.a3s.fi/imigue_skeleton_phase1.zip
https://miga3.a3s.fi/imigue_skeleton_phase2.zip
```
