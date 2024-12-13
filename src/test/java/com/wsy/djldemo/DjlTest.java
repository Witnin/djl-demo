package com.wsy.djldemo;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.junit.jupiter.api.Test;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * <a href="https://www.yuque.com/leifengyang/live/hmfpg92ycirk6r8w#tHi8x">...</a>
 * <a href="https://djl.ai/">...</a>
 * <a href="http://d2l-zh.djl.ai/chapter_preliminaries/index.html">...</a>
 * 完全训练一个模型
 * 1、创建数据集
 * 2、构建神经网络
 * 3、构建模型（应用上面的神经网络）
 * 4、训练模型配置（如何训练、训练集、验证集、测试集）
 * 5、保存使用模型
 * 6、加载模型
 * 7、预测（给模型一个新的输入，让他判断是什么）
 * @author wsy
 * &#064;date  2024/12/13 14:07
 */
public class DjlTest {

    // 测试 N维向量
    @Test
    void testNDArray() {
        // NDManager 创建和管理深度学习期间的临时数据。销毁后自动释放所有资源
        try (
                NDManager manager = NDManager.newBaseManager()) {
            //创建 2x2 矩阵
            /*
              ND: (2, 2) cpu() float32
              [[0., 1.],
               [2., 3.],
              ]
             */
            NDArray ndArray = manager.create(new float[]{0f, 1f, 2f, 3f}, new Shape(2, 2));
            System.out.println("ndArray" + ndArray);


            /*
                2*3的矩阵
              ND: (2, 3) cpu() float32
              [[1., 1., 1.],
               [1., 1., 1.],
              ]
             */
            NDArray ones = manager.ones(new Shape(2, 3));
            System.out.println("ones" + ones);

            /*
              ND: (2, 3, 4) cpu() float32
              [[[0.5488, 0.5928, 0.7152, 0.8443],
                [0.6028, 0.8579, 0.5449, 0.8473],
                [0.4237, 0.6236, 0.6459, 0.3844],
               ],
               [[0.4376, 0.2975, 0.8918, 0.0567],
                [0.9637, 0.2727, 0.3834, 0.4777],
                [0.7917, 0.8122, 0.5289, 0.48  ],
               ],
              ]
             */
            NDArray uniform = manager.randomUniform(0, 1, new Shape(2, 3, 4));
            System.out.println(uniform);
        }
    }

    @Test
    void operation(){
        try(NDManager manager = NDManager.newBaseManager()){
            var array1 = manager. create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
            var array2 = manager. create(new float[] {1f, 2f});

            /*
              ND: (2) cpu() float32
              [ 5., 11.]
             */
            var array3 = array1.matMul(array2);
            System.out.println(array3);

            /*
              ND: (1, 4) cpu() float32
              [[1., 2., 3., 4.],
              ]
             */
            var array4 = array1.reshape(1, 4);
            System.out.println(array4);


            /*
              ND: (2, 2) cpu() float32
              [[1., 3.],
               [2., 4.],
              ]
             */
            var array5 = array1.transpose();
            System.out.println(array5);
        }
    }
    @Test
    void test() throws IOException {
        //1、准备数据集
        RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
        RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST);
        //2、构建神经网络
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[]{128, 64});
        //3、构建模型（应用上面的神经网络）
        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);
            //4、训练模型配置（如何训练、训练集、验证集、测试集）
            DefaultTrainingConfig config = setupTrainingConfig();

            //4、拿到训练器
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                //5、开始训练
                EasyTrain.fit(trainer, 5, trainingSet, validateSet);

                //6、训练结果
                TrainingResult result = trainer.getTrainingResult();
                System.out.println("训练结果:result = " + result);

                //7、保存使用模型
                Path modelDir = Paths.get("build/mlpx");
                Files.createDirectories(modelDir);
// Save the model
                model.save(modelDir, "mlpx");
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }


        }
    }
    //测试模型
    @Test
    void predict() throws IOException, MalformedModelException, TranslateException {
        //8、准备测试数据
        Image image = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");

        Image image2 = ImageFactory.getInstance().fromFile(Paths.get("build/img/7.png"));

        //9、加载模型
        Path modelDir = Paths.get("build/mlpx");
        Model model = Model.newInstance("mlpx");
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[]{128, 64});
        model.setBlock(block);
        model.load(modelDir);
        //10、预测（给模型一个新的输入，让他判断是什么）
        //如果没有 Translator，非标图像则无法处理
        Translator<Image, Classifications> translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(Mnist.IMAGE_WIDTH, Mnist.IMAGE_HEIGHT))
                        .addTransform(new ToTensor())
                        .optApplySoftmax(true)
                        .build();
        Predictor<Image, Classifications> predictor = model.newPredictor(translator);

        Classifications predict = predictor.predict(image2);

        System.out.println("predict = " + predict);
    }

    private RandomAccessDataset getDataset(Dataset.Usage usage) throws IOException {
        Mnist mnist =
                Mnist.builder()
                        .optUsage(usage)
                        .setSampling(64, true)
                        .build();// Mnist数据集内置的/.djl/cache
        mnist.prepare(new ProgressBar());// 进度条
        return mnist;
    }

    private DefaultTrainingConfig setupTrainingConfig() {
        String outputDir = "build/model";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

}
