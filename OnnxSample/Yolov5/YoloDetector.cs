using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxSample.Yolov5
{
    public class YoloDetector : IDisposable
    {
        private InferenceSession sess = null;
        private Mat imageFloat = null;
        private Mat debugImage = null;
        public float MinConfidence { get; set; }
        public float NmsThresh { get; set; }
        private float maxWH = 4096;
        private Size imgSize = new Size(640, 384);
        private Scalar padColor = new Scalar(114, 114, 114);

        /// <summary>
        /// Initialize
        /// </summary>
        /// <param name="model_path"></param>
        public YoloDetector(string model_path)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;            

            sess = new InferenceSession(model_path, option);

            imageFloat = new Mat();
            debugImage = new Mat();
            MinConfidence = 0.2f;
            NmsThresh = 0.4f;
        }


        public List<Prediction> objectDetection(Mat img, double confidence = 0.4)
        {
            MinConfidence = (float)confidence;
            float ratio = 0.0f;
            Point diff1 = new Point();
            Point diff2 = new Point();
            List<Prediction> obj_list = new List<Prediction>();
            //Image -> Letterbox
            bool isAuto = true;
            if (img.Width <= imgSize.Width || img.Height <= imgSize.Height) isAuto = false;
            using (var letterimg = CreateLetterbox(img, imgSize, padColor, out ratio, out diff1, out diff2, auto: isAuto, scaleFill: !isAuto))
            {
                letterimg.ConvertTo(imageFloat, MatType.CV_32FC3, (float)(1 / 255.0));
                var input = new DenseTensor<float>(MatToList(imageFloat), new[] { 1, 3, imgSize.Height, imgSize.Width });
                // Setup inputs and outputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("images", input)
                };

                using (var results = sess.Run(inputs))
                {
                    //Postprocessing
                    var resultsArray = results.ToArray();
                    var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();

                    var nc = pred_dim[pred_dim.Length - 1] - 5;
                    var candidate = GetCandidate(pred_value, pred_dim, MinConfidence);
                    //Compute conf
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var obj_cnf = candidate[i][4];
                        for (int j = 5; j < candidate[i].Count; j++)
                        {
                            candidate[i][j] *= obj_cnf;
                        }
                    }

                    //Change Box coord (xywh -> xyxy)
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var xmin = candidate[i][0] - candidate[i][2] / 2; //top left x
                        var ymin = candidate[i][1] - candidate[i][3] / 2; //top left y
                        var xmax = candidate[i][0] + candidate[i][2] / 2; //bottom right x
                        var ymax = candidate[i][1] + candidate[i][3] / 2; //bottom right y
                        candidate[i][0] = xmin;
                        candidate[i][1] = ymin;
                        candidate[i][2] = xmax;
                        candidate[i][3] = ymax;
                    }
                    //Detections matrix
                    var detected_mat = GetDetectionMatrix(candidate, MinConfidence);
                    //NMS
                    List<Rect> bboxes = new List<Rect>();
                    List<float> confidences = new List<float>();
                    for (int i = 0; i < detected_mat.Count; i++)
                    {
                        var diff_class = (int)(maxWH * detected_mat[i][5]);

                        Rect box = new Rect((int)detected_mat[i][0] + diff_class, (int)detected_mat[i][1] + diff_class,
                            (int)(detected_mat[i][2] - detected_mat[i][0]), (int)(detected_mat[i][3] - detected_mat[i][1]));
                        bboxes.Add(box);
                        confidences.Add(detected_mat[i][4]);
                    }
                    int[] indices = null;
                    CvDnn.NMSBoxes(bboxes, confidences, MinConfidence, NmsThresh, out indices);

                    var predictions = new List<Prediction>();
                    if (indices != null)
                    {
                        for (int ids = 0; ids < indices.Length; ids++)
                        {
                            int idx = indices[ids];
                            var cls = detected_mat[idx][detected_mat[idx].Count - 1];
                            var confi = detected_mat[idx][4];
                            predictions.Add(new Prediction
                            {
                                Box = new Box {
                                    Xmin = detected_mat[idx][0],
                                    Ymin = detected_mat[idx][1],
                                    Xmax = detected_mat[idx][2],
                                    Ymax = detected_mat[idx][3] },
                                Label = LabelMap.Labels[(int)cls],
                                Id = (int)cls,
                                Confidence = confi
                            });
                        }
                    }
                    //Rescale Predictions
                    var rescale_predictions = new List<Prediction>();
                    for (int ids = 0; ids < predictions.Count; ids++)
                    {
                        var pred = predictions[ids];
                        var rescaleBox = CalcRescaleBox(pred.Box, img.Size(), imgSize, diff1, diff2);
                        rescale_predictions.Add(new Prediction
                        {
                            Box = rescaleBox,
                            Label = pred.Label,
                            Id = pred.Id,
                            Confidence = pred.Confidence
                        });
                    }
                    return rescale_predictions;
                }
            }
        }

        public void Dispose()
        {
            debugImage?.Dispose();
            debugImage = null;
            imageFloat?.Dispose();
            imageFloat = null;
            sess?.Dispose();
            sess = null;
        }

        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[ih * iw * chn];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }

        public static Box CalcRescaleBox(Box dBox, Size orgImage, Size resizeImage, Point diff1, Point diff2)
        {
            Box rescaleBox = new Box {
                Xmin = 0,
                Ymin = 0,
                Xmax = 0,
                Ymax = 0
            };
            Point rImgStart = new Point(diff1.X + diff2.X, diff1.Y + diff2.Y);
            Point rImgEnd = new Point(resizeImage.Width - rImgStart.X, resizeImage.Height - rImgStart.Y);

            var ratio_x = orgImage.Width / (float)(rImgEnd.X - rImgStart.X);
            var ratio_y = orgImage.Height / (float)(rImgEnd.Y - rImgStart.Y);
            rescaleBox.Xmin = ratio_x * (dBox.Xmin - rImgStart.X);
            rescaleBox.Xmax = ratio_x * (dBox.Xmax - rImgStart.X);
            rescaleBox.Ymin = ratio_y * (dBox.Ymin - rImgStart.Y);
            rescaleBox.Ymax = ratio_y * (dBox.Ymax - rImgStart.Y);
            return rescaleBox;
        }

        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }

        public static Mat CreateLetterbox(Mat img, Size sz, Scalar color, out float ratio, out Point diff, out Point diff2,
            bool auto = true, bool scaleFill = false, bool scaleup = true)
        {
            Mat newImage = new Mat();
            Cv2.CvtColor(img, newImage, ColorConversionCodes.BGR2RGB);
            ratio = Math.Min((float)sz.Width / newImage.Width, (float)sz.Height / newImage.Height);
            if (!scaleup)
            {
                ratio = Math.Min(ratio, 1.0f);
            }
            var newUnpad = new OpenCvSharp.Size((int)Math.Round(newImage.Width * ratio),
                (int)Math.Round(newImage.Height * ratio));
            var dW = sz.Width - newUnpad.Width;
            var dH = sz.Height - newUnpad.Height;

            var tensor_ratio = sz.Height / (float)sz.Width;
            var input_ratio = img.Height / (float)img.Width;
            if (auto && tensor_ratio != input_ratio)
            {
                dW %= 32;
                dH %= 32;
            }
            else if (scaleFill)
            {
                dW = 0;
                dH = 0;
                newUnpad = sz;
            }
            var dW_h = (int)Math.Round((float)dW / 2);
            var dH_h = (int)Math.Round((float)dH / 2);
            var dw2 = 0;
            var dh2 = 0;
            if (dW_h * 2 != dW)
            {
                dw2 = dW - dW_h * 2;
            }
            if (dH_h * 2 != dH)
            {
                dh2 = dH - dH_h * 2;
            }

            if (newImage.Width != newUnpad.Width || newImage.Height != newUnpad.Height)
            {
                Cv2.Resize(newImage, newImage, newUnpad);
            }
            Cv2.CopyMakeBorder(newImage, newImage, dH_h + dh2, dH_h, dW_h + dw2, dW_h, BorderTypes.Constant, color);
            diff = new OpenCvSharp.Point(dW_h, dH_h);
            diff2 = new OpenCvSharp.Point(dw2, dh2);
            return newImage;
        }

        public static List<List<float>> GetCandidate(float[] pred, int[] pred_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];
                    if (value > pred_thresh)
                    {
                        List<float> tmp_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            tmp_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(tmp_value);
                    }
                }
            }
            return candidate;
        }

        public static List<List<float>> GetDetectionMatrix(List<List<float>> candidate,
            float pred_thresh = 0.25f, int max_nms = 30000)
        {
            var mat = new List<List<float>>();
            for (int i = 0; i < candidate.Count; i++)
            {
                int cls = -1;
                float max_score = 0;
                for (int j = 5; j < candidate[i].Count; j++)
                {
                    if (candidate[i][j] > pred_thresh && candidate[i][j] >= max_score)
                    {
                        cls = j;
                        max_score = candidate[i][j];
                    }
                }

                if (cls < 0) continue;

                List<float> tmpDetect = new List<float>();
                for (int j = 0; j < 4; j++) tmpDetect.Add(candidate[i][j]); //box
                tmpDetect.Add(candidate[i][cls]);   //class prob
                tmpDetect.Add(cls - 5);             //class
                mat.Add(tmpDetect);
            }

            //max_nms sort
            mat.Sort((a, b) => (a[4] > b[4]) ? -1 : 1);

            if (mat.Count > max_nms)
            {
                mat.RemoveRange(max_nms, mat.Count - max_nms);
            }
            return mat;
        }
    }
}
