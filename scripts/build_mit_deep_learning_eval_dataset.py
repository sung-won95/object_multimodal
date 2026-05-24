from __future__ import annotations

import csv
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALS_ROOT = REPO_ROOT.parent
PROJECTS_ROOT = REPO_ROOT / "artifacts" / "paper_mit_deep_learning" / "projects"
PAGES_ROOT = MATERIALS_ROOT / "data" / "MITDeepLearning" / "pages"
OUTPUT_DIR = REPO_ROOT / "eval" / "mit_deep_learning_stt"
SUITES_DIR = OUTPUT_DIR / "suites"

INDEX_UID = "mit_deep_learning_stt_segments"
VISUAL_INDEX_UID = "mit_deep_learning_stt_visual_entities"
RUN_ID = "mit_deep_learning_stt_seed_v1"
ANNOTATOR_ID = "codex_stt_seed_v1"
LICENSE_NAME = "CC BY-NC-SA 4.0"
SOURCE_NAME = "MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024"


FIELDNAMES = [
    "query_id",
    "split",
    "video_id",
    "lecture_title",
    "project_id",
    "project_dir",
    "query_text",
    "query_text_ko",
    "reference_answer",
    "expected_topic",
    "expected_time_hint",
    "expected_visual_hint",
    "gold_start_time",
    "gold_end_time",
    "gold_timestamp_center",
    "gold_modality",
    "gold_visual_entity",
    "gold_entity_link_note",
    "question_type",
    "gold_segment_id",
    "gold_frame_ids",
    "gold_frame_timestamps",
    "gold_visual_entity_ids",
    "gold_visual_entity_texts",
    "linked_entity_count",
    "frame_count",
    "annotator_id",
    "confidence",
    "source",
    "license",
    "notes",
]


EVAL_SPECS: list[dict[str, str]] = [
    {
        "query_id": "mitdl_l01_gradient_descent_idea",
        "video_id": "mit6_7960f24_lec01_mp4",
        "segment_id": "seg_mit6_7960f24_lec01_mp4_000426",
        "query_text": "What is the basic idea behind gradient descent optimization?",
        "query_text_ko": "gradient descent 최적화의 기본 아이디어는 무엇인가?",
        "expected_topic": "gradient descent optimization idea",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l01_backpropagation_breakthrough",
        "video_id": "mit6_7960f24_lec01_mp4",
        "segment_id": "seg_mit6_7960f24_lec01_mp4_000285",
        "query_text": "Which breakthrough concept is introduced in the slide about back propagation?",
        "query_text_ko": "back propagation 슬라이드에서 어떤 돌파구 개념이 소개되는가?",
        "expected_topic": "back propagation as a breakthrough",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l02_prediction_ground_truth_loss",
        "video_id": "mit6_7960f24_lec02_mp4",
        "segment_id": "seg_mit6_7960f24_lec02_mp4_000022",
        "query_text": "How is loss related to the model prediction and the ground truth?",
        "query_text_ko": "loss는 모델 예측과 ground truth의 차이와 어떻게 연결되는가?",
        "expected_topic": "loss from prediction versus ground truth",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l02_pytorch_autograd",
        "video_id": "mit6_7960f24_lec02_mp4",
        "segment_id": "seg_mit6_7960f24_lec02_mp4_000177",
        "query_text": "Why do common operations have defined gradients in PyTorch?",
        "query_text_ko": "PyTorch에서 일반 연산들이 gradient를 갖는 이유는 무엇인가?",
        "expected_topic": "PyTorch autograd",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l03_single_layer_arbitrary_function",
        "video_id": "mit6_7960f24_lec03_mp4",
        "segment_id": "seg_mit6_7960f24_lec03_mp4_000326",
        "query_text": "What can a single layer approximate with enough value units?",
        "query_text_ko": "충분한 value unit이 있으면 single layer가 무엇을 근사할 수 있는가?",
        "expected_topic": "single layer arbitrary function approximation",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l03_exponentially_more_neurons",
        "video_id": "mit6_7960f24_lec03_mp4",
        "segment_id": "seg_mit6_7960f24_lec03_mp4_000920",
        "query_text": "When would the network need exponentially more neurons?",
        "query_text_ko": "어떤 경우에 네트워크가 지수적으로 더 많은 neuron을 필요로 하는가?",
        "expected_topic": "exponentially more neurons",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l04_horizontal_filter",
        "video_id": "mit6_7960f24_lec04_mp4",
        "segment_id": "seg_mit6_7960f24_lec04_mp4_000385",
        "query_text": "What is the filter in the convolution example finding?",
        "query_text_ko": "convolution 예시에서 filter는 무엇을 찾고 있는가?",
        "expected_topic": "horizontal filter in convolution",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l04_residual_connection_same_size",
        "video_id": "mit6_7960f24_lec04_mp4",
        "segment_id": "seg_mit6_7960f24_lec04_mp4_001158",
        "query_text": "What size condition is assumed when adding a residual connection?",
        "query_text_ko": "residual connection을 더할 때 어떤 size 조건을 가정하는가?",
        "expected_topic": "residual connection same size addition",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l05_permutation_invariance",
        "video_id": "mit6_7960f24_lec05_mp4",
        "segment_id": "seg_mit6_7960f24_lec05_mp4_000426",
        "query_text": "What property is called permutation invariance in graph neural networks?",
        "query_text_ko": "graph neural network에서 permutation invariance라고 부르는 속성은 무엇인가?",
        "expected_topic": "permutation invariance",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l05_update_node_representation",
        "video_id": "mit6_7960f24_lec05_mp4",
        "segment_id": "seg_mit6_7960f24_lec05_mp4_000533",
        "query_text": "How does the graph network update a node representation using neighbors?",
        "query_text_ko": "graph network는 이웃을 이용해 node representation을 어떻게 update하는가?",
        "expected_topic": "node representation update from neighbors",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l06_population_risk",
        "video_id": "mit6_7960f24_lec06_mp4",
        "segment_id": "seg_mit6_7960f24_lec06_mp4_000059",
        "query_text": "What does population risk measure on new samples from the world?",
        "query_text_ko": "population risk는 세상에서 온 새 sample에 대해 무엇을 측정하는가?",
        "expected_topic": "population risk on new samples",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l06_occam_generalization",
        "video_id": "mit6_7960f24_lec06_mp4",
        "segment_id": "seg_mit6_7960f24_lec06_mp4_000492",
        "query_text": "Where does most generalization theory come from according to the lecture?",
        "query_text_ko": "강의에 따르면 generalization theory의 대부분은 어디에서 오는가?",
        "expected_topic": "Occam and generalization theory",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l07_learning_rate_step_size",
        "video_id": "mit6_7960f24_lec07_mp4",
        "segment_id": "seg_mit6_7960f24_lec07_mp4_000104",
        "query_text": "What is another name for the learning rate in the optimization update?",
        "query_text_ko": "optimization update에서 learning rate의 다른 이름은 무엇인가?",
        "expected_topic": "learning rate or step size",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l07_steepest_descent",
        "video_id": "mit6_7960f24_lec07_mp4",
        "segment_id": "seg_mit6_7960f24_lec07_mp4_000919",
        "query_text": "What descent method is derived in the visual example?",
        "query_text_ko": "시각 예시에서 어떤 descent 방법을 유도하는가?",
        "expected_topic": "steepest descent",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l08_tokens_array_not_neurons",
        "video_id": "mit6_7960f24_lec08_mp4",
        "segment_id": "seg_mit6_7960f24_lec08_mp4_000188",
        "query_text": "How are tokens described as a basic data structure instead of neurons?",
        "query_text_ko": "token은 neuron 대신 어떤 기본 data structure로 설명되는가?",
        "expected_topic": "array of tokens rather than neurons",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l08_self_attention_weights",
        "video_id": "mit6_7960f24_lec08_mp4",
        "segment_id": "seg_mit6_7960f24_lec08_mp4_000838",
        "query_text": "In self-attention, what determines the weights of the linear combination?",
        "query_text_ko": "self-attention에서 linear combination의 weight는 무엇에 의해 결정되는가?",
        "expected_topic": "self-attention weights from input sequence",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l09_rms_layer_norm",
        "video_id": "mit6_7960f24_lec09_mp4",
        "segment_id": "seg_mit6_7960f24_lec09_mp4_000393",
        "query_text": "Why might RMS norm of activations be used in learning dynamics?",
        "query_text_ko": "learning dynamics에서 activation의 RMS norm을 왜 사용할 수 있는가?",
        "expected_topic": "RMS norm and layer norm",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l09_tune_learning_rate_batch_size",
        "video_id": "mit6_7960f24_lec09_mp4",
        "segment_id": "seg_mit6_7960f24_lec09_mp4_001488",
        "query_text": "Which two training hyperparameters should you always tune?",
        "query_text_ko": "항상 tune해야 하는 두 가지 training hyperparameter는 무엇인가?",
        "expected_topic": "tune learning rate and batch size",
        "question_type": "audio_reasoning",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l10_rnn_concept",
        "video_id": "mit6_7960f24_lec10_mp4",
        "segment_id": "seg_mit6_7960f24_lec10_mp4_000020",
        "query_text": "What does RNN stand for in the memory architectures lecture?",
        "query_text_ko": "memory architectures 강의에서 RNN은 무엇의 약자인가?",
        "expected_topic": "RNN recurrent neural network",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l10_previous_hidden_unit",
        "video_id": "mit6_7960f24_lec10_mp4",
        "segment_id": "seg_mit6_7960f24_lec10_mp4_000302",
        "query_text": "What gets passed from the previous hidden unit into the next one?",
        "query_text_ko": "이전 hidden unit에서 다음 hidden unit으로 무엇이 전달되는가?",
        "expected_topic": "previous hidden unit passed forward",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l11_output_embedding_representation",
        "video_id": "mit6_7960f24_lec11_mp4",
        "segment_id": "seg_mit6_7960f24_lec11_mp4_000038",
        "query_text": "What does the lecture call the output at the top of the representation stack?",
        "query_text_ko": "representation stack의 top output을 강의에서는 무엇이라고 부르는가?",
        "expected_topic": "output embedding or representation",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l11_fine_tuning_idea",
        "video_id": "mit6_7960f24_lec11_mp4",
        "segment_id": "seg_mit6_7960f24_lec11_mp4_000633",
        "query_text": "What is the basic idea of fine-tuning?",
        "query_text_ko": "fine-tuning의 기본 아이디어는 무엇인가?",
        "expected_topic": "keep training as fine-tuning",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l12_contrastive_metric_learning",
        "video_id": "mit6_7960f24_lec12_mp4",
        "segment_id": "seg_mit6_7960f24_lec12_mp4_000013",
        "query_text": "Which ideas are introduced before contrastive representation learning?",
        "query_text_ko": "contrastive representation learning 전에 어떤 아이디어들이 소개되는가?",
        "expected_topic": "metric learning and contrastive representation learning",
        "question_type": "audio_reasoning",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l12_simclr_example",
        "video_id": "mit6_7960f24_lec12_mp4",
        "segment_id": "seg_mit6_7960f24_lec12_mp4_000707",
        "query_text": "Which contrastive learning example is mentioned on the slide?",
        "query_text_ko": "슬라이드에서 어떤 contrastive learning 예시가 언급되는가?",
        "expected_topic": "simCLR example",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l13_representations",
        "video_id": "mit6_7960f24_lec13_mp4",
        "segment_id": "seg_mit6_7960f24_lec13_mp4_000066",
        "query_text": "What are described as good ways to learn representations?",
        "query_text_ko": "representation을 배우는 좋은 방법으로 무엇이 설명되는가?",
        "expected_topic": "ways to learn representations",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l13_kernel_methods",
        "video_id": "mit6_7960f24_lec13_mp4",
        "segment_id": "seg_mit6_7960f24_lec13_mp4_000145",
        "query_text": "What classic example is used when discussing kernel methods?",
        "query_text_ko": "kernel method를 설명할 때 어떤 고전적 예시가 사용되는가?",
        "expected_topic": "classic kernel methods",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l14_generative_model_definition",
        "video_id": "mit6_7960f24_lec14_mp4",
        "segment_id": "seg_mit6_7960f24_lec14_mp4_000076",
        "query_text": "What definition question does the lecture ask about generative models?",
        "query_text_ko": "generative model에 대해 강의가 묻는 definition 질문은 무엇인가?",
        "expected_topic": "definition of a generative model",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l14_diffusion_model_type",
        "video_id": "mit6_7960f24_lec14_mp4",
        "segment_id": "seg_mit6_7960f24_lec14_mp4_000107",
        "query_text": "What type of generative model is shown in the diffusion example?",
        "query_text_ko": "diffusion 예시에서 어떤 유형의 generative model이 보이는가?",
        "expected_topic": "diffusion model as generative model",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l15_z_to_x_x_to_z",
        "video_id": "mit6_7960f24_lec15_mp4",
        "segment_id": "seg_mit6_7960f24_lec15_mp4_000074",
        "query_text": "What are the mappings from Z to X and from X to Z called in the VAE setup?",
        "query_text_ko": "VAE 설정에서 Z에서 X로, X에서 Z로 가는 mapping은 어떻게 설명되는가?",
        "expected_topic": "Z to X and X to Z mappings",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l15_marginalizing_over_z",
        "video_id": "mit6_7960f24_lec15_mp4",
        "segment_id": "seg_mit6_7960f24_lec15_mp4_000334",
        "query_text": "How is the total probability expressed with respect to Z?",
        "query_text_ko": "Z에 대해 total probability는 어떻게 표현되는가?",
        "expected_topic": "marginalizing over Z",
        "question_type": "formula_table_lookup",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l16_conditional_setting",
        "video_id": "mit6_7960f24_lec16_mp4",
        "segment_id": "seg_mit6_7960f24_lec16_mp4_000512",
        "query_text": "What setting does the lecture revisit for conditional generative models?",
        "query_text_ko": "conditional generative model에서 강의는 어떤 setting을 다시 살펴보는가?",
        "expected_topic": "conditional setting",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l16_multimodal_representation",
        "video_id": "mit6_7960f24_lec16_mp4",
        "segment_id": "seg_mit6_7960f24_lec16_mp4_001067",
        "query_text": "What is integrated into a multimodal representation of words and images?",
        "query_text_ko": "word와 image의 multimodal representation에는 무엇이 통합되는가?",
        "expected_topic": "multimodal representation of words and images",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l17_ood_generalization",
        "video_id": "mit6_7960f24_lec17_mp4",
        "segment_id": "seg_mit6_7960f24_lec17_mp4_000004",
        "query_text": "What is out-of-distribution generalization about in this lecture?",
        "query_text_ko": "이 강의에서 out-of-distribution generalization은 무엇에 관한 것인가?",
        "expected_topic": "data not covered by training",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l17_infinity_norm_pixels",
        "video_id": "mit6_7960f24_lec17_mp4",
        "segment_id": "seg_mit6_7960f24_lec17_mp4_000357",
        "query_text": "What does the infinity norm constrain for adversarial pixel changes?",
        "query_text_ko": "adversarial pixel 변화에서 infinity norm은 무엇을 제한하는가?",
        "expected_topic": "infinity norm maximum pixel change",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l18_transfer_learning_high_level",
        "video_id": "mit6_7960f24_lec18_mp4",
        "segment_id": "seg_mit6_7960f24_lec18_mp4_000007",
        "query_text": "At a high level, what is the idea of transfer learning?",
        "query_text_ko": "high level에서 transfer learning의 아이디어는 무엇인가?",
        "expected_topic": "high-level transfer learning",
        "question_type": "concept_explanation",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l18_pretraining_task_a",
        "video_id": "mit6_7960f24_lec18_mp4",
        "segment_id": "seg_mit6_7960f24_lec18_mp4_000201",
        "query_text": "What happens to the network during pre-training on task A?",
        "query_text_ko": "task A에서 pre-training할 때 network에는 어떤 일이 일어나는가?",
        "expected_topic": "pre-training a network on task A",
        "question_type": "audio_reasoning",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_l19_generator_system",
        "video_id": "mit6_7960f24_lec19_mp4",
        "segment_id": "seg_mit6_7960f24_lec19_mp4_000072",
        "query_text": "What does the data plus generator system let us define operators over?",
        "query_text_ko": "data plus generator system은 무엇에 대한 operator를 정의하게 해주는가?",
        "expected_topic": "operators over data plus generator",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l19_pretraining_worse_than",
        "video_id": "mit6_7960f24_lec19_mp4",
        "segment_id": "seg_mit6_7960f24_lec19_mp4_001154",
        "query_text": "What can sometimes be worse than pre-training on a broad dataset?",
        "query_text_ko": "넓은 dataset에서 pre-training하는 것보다 때로 무엇이 더 나쁠 수 있는가?",
        "expected_topic": "sometimes worse than pre-training",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l20_data_compute_parameters",
        "video_id": "mit6_7960f24_lec20_mp4",
        "segment_id": "seg_mit6_7960f24_lec20_mp4_000011",
        "query_text": "When does deep learning work really well according to the scaling laws lecture?",
        "query_text_ko": "scaling laws 강의에 따르면 deep learning은 언제 특히 잘 작동하는가?",
        "expected_topic": "data compute parameters",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l20_attention_length_plot",
        "video_id": "mit6_7960f24_lec20_mp4",
        "segment_id": "seg_mit6_7960f24_lec20_mp4_000426",
        "query_text": "What question is asked about attention length and the plot?",
        "query_text_ko": "attention length와 plot의 관계에 대해 어떤 질문이 제기되는가?",
        "expected_topic": "attention length relation to plot",
        "question_type": "visual_object_reference",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l21_word_distribution_corpus",
        "video_id": "mit6_7960f24_lec21_mp4",
        "segment_id": "seg_mit6_7960f24_lec21_mp4_000120",
        "query_text": "How is the word distribution approximated in the corpus example?",
        "query_text_ko": "corpus 예시에서 word distribution은 어떻게 근사되는가?",
        "expected_topic": "counting words in a corpus",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l21_in_context_hypothesis",
        "video_id": "mit6_7960f24_lec21_mp4",
        "segment_id": "seg_mit6_7960f24_lec21_mp4_000573",
        "query_text": "What hypothesis is raised about in-context learning?",
        "query_text_ko": "in-context learning에 대해 어떤 hypothesis가 제기되는가?",
        "expected_topic": "hypothesis about in-context learning",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l23_second_order_loss",
        "video_id": "mit6_7960f24_lec23_mp4",
        "segment_id": "seg_mit6_7960f24_lec23_mp4_000217",
        "query_text": "What term in the loss is decomposed in the metrized deep learning lecture?",
        "query_text_ko": "metrized deep learning 강의에서 loss의 어떤 항을 분해하는가?",
        "expected_topic": "second order term in the loss",
        "question_type": "formula_table_lookup",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l23_compounds_combinations",
        "video_id": "mit6_7960f24_lec23_mp4",
        "segment_id": "seg_mit6_7960f24_lec23_mp4_000512",
        "query_text": "What name is given to things built by combinations?",
        "query_text_ko": "combination으로 만들어지는 것들에 어떤 이름을 붙이는가?",
        "expected_topic": "compounds built by combinations",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l24_greedy_sampling_next_token",
        "video_id": "mit6_7960f24_lec24_mp4",
        "segment_id": "seg_mit6_7960f24_lec24_mp4_000401",
        "query_text": "What does greedy sampling do with the most likely next token?",
        "query_text_ko": "greedy sampling은 가장 가능성 높은 next token을 어떻게 사용하는가?",
        "expected_topic": "greedy sampling most likely next token",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_l24_rlhf_preferences",
        "video_id": "mit6_7960f24_lec24_mp4",
        "segment_id": "seg_mit6_7960f24_lec24_mp4_000824",
        "query_text": "What method is described as moving toward human preferences?",
        "query_text_ko": "human preference 쪽으로 이동하는 방법으로 무엇이 설명되는가?",
        "expected_topic": "RLHF human preferences",
        "question_type": "definition_lookup",
        "gold_modality": "audio",
    },
    {
        "query_id": "mitdl_pytorch_autograd_gradients",
        "video_id": "mit6_7960f24_review_mp4",
        "segment_id": "seg_mit6_7960f24_review_mp4_000020",
        "query_text": "What is AutoGrad used to compute in PyTorch?",
        "query_text_ko": "PyTorch에서 AutoGrad는 무엇을 계산하는 데 쓰이는가?",
        "expected_topic": "AutoGrad computes gradients",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
    {
        "query_id": "mitdl_pytorch_gradient_descent_update",
        "video_id": "mit6_7960f24_review_mp4",
        "segment_id": "seg_mit6_7960f24_review_mp4_000437",
        "query_text": "How are parameters updated in the PyTorch neural network tutorial?",
        "query_text_ko": "PyTorch neural network tutorial에서 parameter는 어떻게 update되는가?",
        "expected_topic": "gradient descent updates parameters",
        "question_type": "multimodal_grounded",
        "gold_modality": "both",
    },
]


def main() -> None:
    projects = _load_projects()
    titles = _load_titles()
    rows = [_build_row(index, spec, projects, titles) for index, spec in enumerate(EVAL_SPECS)]
    _write_outputs(rows, projects)
    _validate_rows(rows)
    print(json.dumps(_summary(rows), ensure_ascii=False, indent=2))


def _load_projects() -> dict[str, dict[str, Any]]:
    projects: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(PROJECTS_ROOT.glob("*/manifests/project_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_id = str(manifest["video_id"])
        project_dir = manifest_path.parents[1]
        segments = _read_jsonl(project_dir / "segments" / "lecture_segments_aligned.jsonl")
        frames = _read_jsonl(project_dir / "manifests" / "frames_manifest.jsonl")
        visual_entities = _read_jsonl(project_dir / "manifests" / "visual_entities.jsonl")
        entity_links = _read_jsonl(project_dir / "manifests" / "entity_links.jsonl")
        projects[video_id] = {
            "manifest": manifest,
            "project_dir": project_dir,
            "segments": {str(item["segment_id"]): item for item in segments},
            "segments_ordered": segments,
            "frames": {str(item["frame_id"]): item for item in frames},
            "visual_entities": {str(item["entity_id"]): item for item in visual_entities},
            "links_by_segment": _group_links(entity_links),
        }
    return projects


def _load_titles() -> dict[str, str]:
    titles: dict[str, str] = {}
    for page_path in sorted(PAGES_ROOT.glob("mit6_7960f24_*.html")):
        text = page_path.read_text(encoding="utf-8", errors="ignore")
        title_match = re.search(r"<title>(.*?) \| Deep Learning", text)
        if title_match:
            titles[page_path.stem] = html.unescape(title_match.group(1)).strip()
    return titles


def _build_row(
    index: int,
    spec: dict[str, str],
    projects: dict[str, dict[str, Any]],
    titles: dict[str, str],
) -> dict[str, str]:
    video_id = spec["video_id"]
    if video_id not in projects:
        raise ValueError(f"Missing project for video_id={video_id}")
    project = projects[video_id]
    segment_id = spec["segment_id"]
    segment = project["segments"].get(segment_id)
    if segment is None:
        raise ValueError(f"Missing segment {segment_id} in {video_id}")

    start = _float(segment.get("start_time"))
    end = _float(segment.get("end_time"))
    center = _float(segment.get("timestamp_center")) or ((start + end) / 2.0)
    frame_ids = [str(item) for item in segment.get("frame_refs") or []]
    frame_times = [
        _format_seconds(project["frames"][frame_id].get("timestamp"))
        for frame_id in frame_ids
        if frame_id in project["frames"]
    ]
    links = sorted(
        project["links_by_segment"].get(segment_id, []),
        key=lambda item: _float(item.get("score")),
        reverse=True,
    )
    visual_entity_ids, visual_entity_texts = _linked_visual_entities(
        links=links,
        visual_entities=project["visual_entities"],
    )
    visual_hint = _visual_hint(frame_ids, frame_times, visual_entity_texts)
    note_parts = [
        "seed label generated from STT-aligned segment artifacts",
        "human audit recommended before final paper claims",
    ]
    if spec.get("notes"):
        note_parts.append(spec["notes"])

    return {
        "query_id": spec["query_id"],
        "split": "dev" if index % 2 == 0 else "test",
        "video_id": video_id,
        "lecture_title": titles.get(video_id, video_id),
        "project_id": str(project["manifest"]["project_id"]),
        "project_dir": _rel(project["project_dir"]),
        "query_text": spec["query_text"],
        "query_text_ko": spec["query_text_ko"],
        "reference_answer": _reference_answer(project["segments_ordered"], segment_id),
        "expected_topic": spec["expected_topic"],
        "expected_time_hint": f"{_format_seconds(start)}-{_format_seconds(end)}s",
        "expected_visual_hint": visual_hint,
        "gold_start_time": _format_seconds(start),
        "gold_end_time": _format_seconds(end),
        "gold_timestamp_center": _format_seconds(center),
        "gold_modality": spec["gold_modality"],
        "gold_visual_entity": visual_hint if spec["gold_modality"] != "audio" else "",
        "gold_entity_link_note": _entity_link_note(
            spec=spec,
            frame_ids=frame_ids,
            visual_entity_texts=visual_entity_texts,
        ),
        "question_type": spec["question_type"],
        "gold_segment_id": segment_id,
        "gold_frame_ids": "|".join(frame_ids),
        "gold_frame_timestamps": "|".join(frame_times),
        "gold_visual_entity_ids": "|".join(visual_entity_ids),
        "gold_visual_entity_texts": "|".join(visual_entity_texts),
        "linked_entity_count": str(len(links)),
        "frame_count": str(len(frame_ids)),
        "annotator_id": ANNOTATOR_ID,
        "confidence": "high" if spec["gold_modality"] == "audio" else "medium",
        "source": SOURCE_NAME,
        "license": LICENSE_NAME,
        "notes": "; ".join(note_parts),
    }


def _write_outputs(rows: list[dict[str, str]], projects: dict[str, dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "queries.csv", rows)
    _write_jsonl(OUTPUT_DIR / "queries.jsonl", rows)

    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_id"]].append(row)
    for video_id, video_rows in rows_by_video.items():
        _write_csv(SUITES_DIR / f"{video_id}.csv", video_rows)

    manifest = {
        "run_id": RUN_ID,
        "output_dir": "../../reports/mit_deep_learning_eval/seed_v1",
        "deltas": [5, 10, 15, 30],
        "suites": [
            {
                "suite_id": _suite_id(video_id),
                "type": "local_project",
                "domain": "mit_deep_learning_stt_seed",
                "project_dir": _rel(projects[video_id]["project_dir"], base=OUTPUT_DIR),
                "queries": f"suites/{video_id}.csv",
                "index": INDEX_UID,
                "visual_index": VISUAL_INDEX_UID,
                "video_id": video_id,
                "limit": 5,
                "neighbor_count": 1,
                "rerank": False,
            }
            for video_id in sorted(rows_by_video)
        ],
    }
    (OUTPUT_DIR / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(_summary(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(_readme(rows), encoding="utf-8")


def _validate_rows(rows: list[dict[str, str]]) -> None:
    seen = set()
    for row in rows:
        if row["query_id"] in seen:
            raise ValueError(f"Duplicate query_id={row['query_id']}")
        seen.add(row["query_id"])
        if not row["query_text"] or not row["reference_answer"]:
            raise ValueError(f"Missing query/reference answer for {row['query_id']}")
        if _float(row["gold_end_time"]) < _float(row["gold_start_time"]):
            raise ValueError(f"Invalid time range for {row['query_id']}")
        if row["gold_modality"] in {"both", "vision"} and not row["gold_frame_ids"]:
            raise ValueError(f"Missing frame evidence for {row['query_id']}")


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "query_count": len(rows),
        "video_count": len({row["video_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "modality_counts": dict(sorted(Counter(row["gold_modality"] for row in rows).items())),
        "question_type_counts": dict(
            sorted(Counter(row["question_type"] for row in rows).items())
        ),
        "frame_backed_count": sum(1 for row in rows if row["gold_frame_ids"]),
        "linked_entity_backed_count": sum(
            1 for row in rows if int(row["linked_entity_count"]) > 0
        ),
        "index": INDEX_UID,
        "visual_index": VISUAL_INDEX_UID,
        "license": LICENSE_NAME,
        "source": SOURCE_NAME,
        "missing_lecture_note": "Lecture 22 is absent from the MIT OCW gallery and from the local corpus.",
    }


def _readme(rows: list[dict[str, str]]) -> str:
    summary = _summary(rows)
    return "\n".join(
        [
            "# MIT Deep Learning STT Seed Evaluation Set",
            "",
            "이 폴더는 STT로 구축한 MIT 6.7960 Deep Learning RAG를 평가하기 위한 1차 seed 평가셋이다.",
            "각 row는 질문, gold segment timestamp, modality label, frame/link evidence 힌트를 포함한다.",
            "",
            "## Files",
            "",
            "- `queries.csv`: 통합 평가셋",
            "- `queries.jsonl`: 동일 내용을 JSONL로 저장한 버전",
            "- `suites/*.csv`: `benchmark-retrieval`이 강의별로 읽는 CSV",
            "- `benchmark_manifest.json`: 바로 실행 가능한 retrieval benchmark manifest",
            "- `summary.json`: row count, split, modality 분포 요약",
            "",
            "## Scope",
            "",
            f"- Queries: {summary['query_count']}",
            f"- Videos: {summary['video_count']}",
            f"- Splits: {summary['split_counts']}",
            f"- Modalities: {summary['modality_counts']}",
            f"- Question types: {summary['question_type_counts']}",
            "",
            "## Use",
            "",
            "```bash",
            "python -m oarag benchmark-retrieval \\",
            "  --manifest eval/mit_deep_learning_stt/benchmark_manifest.json",
            "```",
            "",
            "## Caveat",
            "",
            "이 데이터셋은 STT segment와 대표 frame alignment에서 만든 seed label이다.",
            "정식 논문 수치로 쓰기 전에는 사람이 timestamp, modality, visual entity를 한 번 더 검수하는 것이 좋다.",
            "",
            f"Source: {SOURCE_NAME}",
            f"License: {LICENSE_NAME}",
            "",
        ]
    )


def _group_links(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["segment_id"])].append(row)
    return grouped


def _linked_visual_entities(
    *,
    links: list[dict[str, Any]],
    visual_entities: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    for link in links:
        entity_id = str(link.get("entity_id") or "")
        entity = visual_entities.get(entity_id)
        if entity is None:
            continue
        text = _clean_visual_text(str(entity.get("text") or entity.get("visual_description") or ""))
        if not text:
            continue
        ids.append(entity_id)
        if text not in texts:
            texts.append(text)
        if len(texts) >= 8:
            break
    return ids[:8], texts[:8]


def _visual_hint(
    frame_ids: list[str],
    frame_times: list[str],
    visual_entity_texts: list[str],
) -> str:
    if not frame_ids:
        return ""
    frame_part = "frames " + "|".join(frame_ids)
    if frame_times:
        frame_part += " at " + "|".join(f"{time}s" for time in frame_times)
    if visual_entity_texts:
        return frame_part + "; OCR cues: " + ", ".join(visual_entity_texts[:6])
    return frame_part


def _entity_link_note(
    *,
    spec: dict[str, str],
    frame_ids: list[str],
    visual_entity_texts: list[str],
) -> str:
    if spec["gold_modality"] == "audio":
        return ""
    query_terms = spec["expected_topic"]
    frame_part = "|".join(frame_ids) if frame_ids else "aligned frame"
    if visual_entity_texts:
        return (
            f"Question topic '{query_terms}' is grounded by STT at the gold segment and "
            f"the aligned frame(s) {frame_part}; OCR/link cues include "
            f"{', '.join(visual_entity_texts[:4])}."
        )
    return (
        f"Question topic '{query_terms}' is grounded by STT at the gold segment and "
        f"the aligned representative frame(s) {frame_part}."
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _reference_answer(segments: list[dict[str, Any]], segment_id: str) -> str:
    for index, segment in enumerate(segments):
        if str(segment.get("segment_id")) != segment_id:
            continue
        window = segments[max(0, index - 1) : min(len(segments), index + 2)]
        return _clean_text(" ".join(str(item.get("transcript_text") or "") for item in window))
    return ""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _suite_id(video_id: str) -> str:
    return video_id.replace("mit6_7960f24_", "mitdl_").replace("_mp4", "")


def _rel(path: Path, *, base: Path | None = None) -> str:
    anchor = base or REPO_ROOT
    return os.path.relpath(path.resolve(), anchor.resolve())


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_visual_text(value: str) -> str:
    text = _clean_text(value)
    if len(text) < 3:
        return ""
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return text


def _float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _format_seconds(value: Any) -> str:
    return f"{_float(value):.1f}"


if __name__ == "__main__":
    main()
