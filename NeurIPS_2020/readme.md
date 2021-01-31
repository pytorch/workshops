Building AI with Security and Privacy in Mind - NeurIPS 2020 Workshop
========================================================================
*Laurens van der Maaten, Davide Testuggine, Andrew Trask, Geeta Chauhan*

Practical applications of ML via cloud-based or machine-learning-as-a-service platforms pose a range of security and privacy challenges. There are a number of technical approaches being studied including: homomorphic encryption, secure multi-party computation, federated learning, on-device computation, and differential privacy. This tutorial will dive into some of the important areas that are shaping the future of how we interpret our models and build AI with security and privacy in mind. We will cover the major challenges and walk through some solutions. The material will be presented in the following talks:

- Introduction to Privacy Preserving Machine Learning - Geeta Chauhan, [Recording](https://slideslive.com/38942327/opacus-differential-privacy-on-pytorch)
- Secure Computation using [CrypTen](https://crypten.ai/); - Laurens van der Maaten, [Recording](https://slideslive.com/38943021/secure-computation-using-crypten)
- Training models differentially private at scale using [Opacus](https://ai.facebook.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/); - Davide Testuggine, [Recording](https://slideslive.com/38943022/building-ai-with-security-and-privacy-in-mind )
- Training models across multiple organizations privately with federated learning and PySyft from [OpenMined](https://www.openmined.org/) - Andrew Trask, [Recording](https://slideslive.com/38943023/building-ai-with-security-and-privacy-in-mind)

The tutorial will start with basic concepts and will proceed into more advanced topics following a chronological order of the presentations. The audience is expected to have some basic understanding of deep learning frameworks and models that will be supplemented with the material in the early talks. The audience will have an opportunity to learn more advanced topics and models as the tutorial proceeds.

### Details

*PPML*: Practical applications of machine learning via cloud-based or machine-learning-as-a-service (MLaaS) platforms pose a range of security and privacy challenges. In particular, users of these platforms may not be willing or allowed to share their data with platform providers, which prevents them from taking full advantage of the value of the platforms. To address these challenges there are a number of technical approaches being studied at various levels of maturity including: homomorphic encryption, secure multi-party computation, federated learning, on-device computation, and differential privacy. This talk will provide an introduction to Privacy Preserving ML, challenges and solutions. 

*CrypTen:* There are many scenarios in which two or more parties could create a lot of value by combining the data they possess and training machine-learning models on that combined data, but cannot do so because the parties want to keep their data secure and private. Secure multi-party computation (MPC) can facilitate joint training of models while preserving the privacy and security of each party’s data. In this tutorial, we will teach you how to perform such training using the CrypTen platform. CrypTen allows machine-learning researchers to convert their PyTorch code to use secure MPC with minimal changes. You will get hands-on experience on how to quickly build and deploy machine-learning models using CrypTen, and learn the intricacies of the secure MPC protocols that back it. 

*Opacus:* Machine learning models have become increasingly better at memorizing the data they were trained on. This represents a privacy risk which has led to successful attacks [Fredriksen et al 2015] [Carlini et al 2018]. Differential privacy allows us to both measure memorization and keep it at bay. In this session, we will go over the key ideas behind differentially private deep learning, and we will use Opacus in a live demo to train a differentially-private ResNet in a few lines of code.

*Federated Learning:* AI progress is driven by three things: data, compute, and model development. The perennial problem of AI both in research and in production is “not enough data”. In this hands on tutorial, we will teach you how to use PyTorch for Federated Learning on data owned by other researchers, universities, and enterprises - unlocking orders of magnitude more data for your projects and setting you up with a massive competitive advantage in your pursuit of the next great breakthrough in AI.  You will get hands-on experience with state-of-the-art tools for Federated Learning, with broad applicability across every scientific discipline and application area of AI. In short - you will learn how to train and evaluate AI models on data you do not have access to.


### Relevant References
- [Crypten](https://ai.facebook.com/blog/crypten-a-new-research-tool-for-secure-machine-learning-with-pytorch/)
- [Opacus](https://ai.facebook.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/)
- [PySyft paper](https://arxiv.org/abs/1811.04017)
- [OpenMined](https://www.openmined.org/)
