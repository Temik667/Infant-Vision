# Infant Vision
Our team investigated the effect of low visual acuity and colour sensitivity on learning
performance and speed using ResNET18 and Tiny Imagenet Dataset. The model used SDG with
momentum with early stopping, 10 epochs. The transformation code is taken from Ashna Chalil for
color perception and acuity simulation. To observe the effect of transformations, we trained 4 identical
models with identical parameters, but with different datasets:
## Low visual acuity model
In this model, we transformed the training data(13 datasets in total) into pictures with blurry
resolution progressing from 0 to 12 months. Then, the model was trained on each dataset, until the
early stopping was triggered based on the validation value, after which the next dataset with less blur
was used. To be precise, the model trained on a dataset that contained images transformed to 0
months infant’s view and so on, each new dataset decreased the amount of blurring. This was
expected to increase the accuracy and establish effective receptive fields.
## Low colour sensitivity model
In this training, we utilised the same parameters and dataset, however, only the colour
sensitivity was changing. From 0 to 12 months, we mimicked the development of colour perception as
in infants. The model trained on each month's dataset, and as previously mentioned, the early
stopping was implemented to skip to the next dataset, when the model accuracy was saturated.
## Low colour sensitivity and low acuity model
This model is expected to combine the two features that appear in an infant’s vision during its
first 12 months. Therefore, every new dataset in this model contained pictures that progressively
reduce the blurriness and increase the colour contrast. We expected the model to adapt to effectively
use high spatial frequency cues as well as on colour cues.
## No transformation(default) model
This model is a reference model to which we will be comparing the models with the
transformed dataset. The early stopping was implemented also, to speed up the training and
effectively compare models.
# Conclusion
Observing the validation accuracies, we can see that the model that included transform
performs significantly worse. We believe this might be due to the low resolution of the images (64x64
pixels). Since images are already low resolution, objects are heavily blurred. Applying blurring and low
colour sensitivity only hinders learning, since high spatial frequencies are removed and colour cues
are absent, which are the primary cues when objects appear blurred. The importance of color in low
resolution can be seen when comparing low color sensitivity and acuity models, the accuracy for the
latter one is significantly less. We hypothesize that high spatial frequency features play a more critical
role in model perception than high color contrast. However, when color sensitivity is low, acuity
remains unaffected, yet model performance remains significantly impaired. Both low acuity and low
color sensitivity hinder training effectiveness. Nevertheless, high color sensitivity provides less
performance improvement in later stages compared to high acuity. Interestingly, the model can adapt
to high-acuity vision even when it begins with low acuity. In contrast, if the model starts with low color
sensitivity, it struggles to improve performance later, even if color sensitivity increases.
