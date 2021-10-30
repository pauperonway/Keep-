// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private String[] mTestImages = {"test3.png", "test2.jpg", "test1.png"};

    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    static {
//        System.loadLibrary("opencv_java");
        System.loadLibrary("opencv_java3");
    }

    public static boolean isOpenCVInit = false;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/3"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "pose_hrnet_w32_256x192.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

    public float[][] mPrintPointArray = null;
    public static int mNumKeypoint = 17;
    private Mat mMat = null;

    protected int getImageSizeX() {
        return 192;
    }
    protected int getImageSizeY() {
        return 256;
    }

    protected int getOutputSizeX() {
        return 48;
    }
    protected int getOutputSizeY() {
        return 64;
    }
    protected float getRatio(){
        return (float)getOutputSizeX() / (float)getImageSizeX();
    }

    @Override
    public void run() {
        // gb add read mp4 begin
        OpenCVLoader.initDebug();
        String mp4_path = null;
        try {
            mp4_path = MainActivity.assetFilePath(getApplicationContext(), "shooting-model.mp4");
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading mp4.", e);
        }
        //String modelPath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "mnn" + File.separator + "cpm.mnn";
        //Log.i("##############gb333", modelPath);
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        Log.i("##############gb333", mp4_path);
        retriever.setDataSource(mp4_path);
        // 取得视频的长度(单位为毫秒)
        String time = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        // 取得视频的长度(单位为毫秒)
        int ms = Integer.valueOf(time);
        Log.i("##############gb ms: ", String.valueOf(ms));
        // 得到每一秒时刻的bitmap比如第一秒,第二秒
        for (int m = 1; m <= ms; m+=33) {
            Log.i("##############gb m:", String.valueOf(m));
            Bitmap bitmap = retriever.getFrameAtTime(m * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
            //Bitmap bitmap = null;
            //try {
            //    bitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
            //} catch (IOException e) {
            //    Log.e("Object Detection", "Error reading assets", e);
            //    finish();
            //}
            Log.i("##############gb m:", "1111111111111111111111111");
            Log.i("##############gb:", String.valueOf(bitmap.getWidth()));
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
            Log.i("##############gb","222222222222222222222222");
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
            Log.i("##############gb","3333333333333333333333333");
            final Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
            //final Tensor outputTensor = outputTuple[0].toTensor();
            long[] in_size = inputTensor.shape();
            long[] out_size = outputTensor.shape();
            Log.i("##############gb out_size 0#######", String.valueOf(out_size[0]));
            Log.i("##############gb out_size 1#######", String.valueOf(out_size[1]));
            Log.i("##############gb out_size 2#######", String.valueOf(out_size[2]));
            Log.i("##############gb out_size 3#######", String.valueOf(out_size[3]));
            Log.i("##############gb", String.valueOf(bitmap.getHeight()));
            Mat res_mat = new Mat(resizedBitmap.getHeight(), resizedBitmap.getWidth(), CvType.CV_8UC3);
            org.opencv.android.Utils.bitmapToMat(resizedBitmap, res_mat);
            Imgproc.cvtColor(res_mat,res_mat,Imgproc.COLOR_RGB2BGR);
            String im_path = null;
            try {
                im_path = MainActivity.assetFilePath(getApplicationContext(), "test.jpg");
            } catch (IOException e) {
                Log.e("##############gb", "Error save im.", e);
            }


            float[] result = outputTensor.getDataAsFloatArray();
            if (mPrintPointArray == null)
                mPrintPointArray = new float[2][mNumKeypoint];

            //先进行高斯滤波,5*5
            if (mMat == null)
                mMat = new Mat(getOutputSizeY(), getOutputSizeX(), CvType.CV_32F);

            float[] tempArray = new float[getOutputSizeY() * getOutputSizeX()];
            float[] outTempArray = new float[getOutputSizeY() * getOutputSizeX()];

            for (int i = 0; i < mNumKeypoint; i++) {
                int index = 0;
                for (int y = 0; y < getOutputSizeY(); y++) {
                    for (int x = 0; x < getOutputSizeX(); x++) {
                        tempArray[index] = result[i * getOutputSizeY() * getOutputSizeX() + y * getOutputSizeX() + x];
                        index++;
                    }
                }

                mMat.put(0, 0, tempArray);
                Imgproc.GaussianBlur(mMat, mMat, new Size(3, 3), 1, 1);
                mMat.get(0, 0, outTempArray);
                float maxX = 0, maxY = 0;
                float max = 0;

                for (int y = 0; y < getOutputSizeY(); y++) {
                    for (int x = 0; x < getOutputSizeX(); x++) {
                        float center = get(x, y, outTempArray);
                        if (center >= 0.01) {
                            if (center > max) {
                                max = center;
                                maxX = x;
                                maxY = y;
                            }
                        }
                    }
                }
                if (max == 0) {
                    maxX = 0;
                    maxY = 0;
                }

                mPrintPointArray[0][i] = maxX / getRatio();
                mPrintPointArray[1][i] = maxY / getRatio();
                Log.i("##############gb i########", String.valueOf(i));
                Log.i("##############gb maxX########", String.valueOf(mPrintPointArray[0][i]));
                Log.i("##############gb maxY########", String.valueOf(mPrintPointArray[1][i]));
            }

            //0-1
            if (mPrintPointArray[0][0] > 0 && mPrintPointArray[1][0] > 0
                    && mPrintPointArray[0][1] > 0 && mPrintPointArray[1][1] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][0], mPrintPointArray[1][0]),
                        new Point(mPrintPointArray[0][1], mPrintPointArray[1][1]), new Scalar(0, 0, 255), 2);
            }
            //0-2
            if (mPrintPointArray[0][0] > 0 && mPrintPointArray[1][0] > 0
                    && mPrintPointArray[0][2] > 0 && mPrintPointArray[1][2] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][0], mPrintPointArray[1][0]),
                        new Point(mPrintPointArray[0][2], mPrintPointArray[1][2]), new Scalar(0, 0, 255), 2);
            }
            //5-6
            if (mPrintPointArray[0][5] > 0 && mPrintPointArray[1][5] > 0
                    && mPrintPointArray[0][6] > 0 && mPrintPointArray[1][6] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][5], mPrintPointArray[1][5]),
                        new Point(mPrintPointArray[0][6], mPrintPointArray[1][6]), new Scalar(0, 0, 255), 2);
            }
            //5-7
            if (mPrintPointArray[0][5] > 0 && mPrintPointArray[1][5] > 0
                    && mPrintPointArray[0][7] > 0 && mPrintPointArray[1][7] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][5], mPrintPointArray[1][5]),
                        new Point(mPrintPointArray[0][7], mPrintPointArray[1][7]), new Scalar(0, 0, 255), 2);
            }
            //7-9
            if (mPrintPointArray[0][7] > 0 && mPrintPointArray[1][7] > 0
                    && mPrintPointArray[0][9] > 0 && mPrintPointArray[1][9] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][7], mPrintPointArray[1][7]),
                        new Point(mPrintPointArray[0][9], mPrintPointArray[1][9]), new Scalar(0, 0, 255), 2);
            }
            //6-8
            if (mPrintPointArray[0][6] > 0 && mPrintPointArray[1][6] > 0
                    && mPrintPointArray[0][8] > 0 && mPrintPointArray[1][8] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][6], mPrintPointArray[1][6]),
                        new Point(mPrintPointArray[0][8], mPrintPointArray[1][8]), new Scalar(0, 0, 255), 2);
            }
            //8-10
            if (mPrintPointArray[0][8] > 0 && mPrintPointArray[1][8] > 0
                    && mPrintPointArray[0][10] > 0 && mPrintPointArray[1][10] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][8], mPrintPointArray[1][8]),
                        new Point(mPrintPointArray[0][10], mPrintPointArray[1][10]), new Scalar(0, 0, 255), 2);
            }
            //5-11
            if (mPrintPointArray[0][5] > 0 && mPrintPointArray[1][5] > 0
                    && mPrintPointArray[0][11] > 0 && mPrintPointArray[1][11] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][5], mPrintPointArray[1][5]),
                        new Point(mPrintPointArray[0][11], mPrintPointArray[1][11]), new Scalar(0, 0, 255), 2);
            }
            //6-12
            if (mPrintPointArray[0][6] > 0 && mPrintPointArray[1][6] > 0
                    && mPrintPointArray[0][12] > 0 && mPrintPointArray[1][12] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][6], mPrintPointArray[1][6]),
                        new Point(mPrintPointArray[0][12], mPrintPointArray[1][12]), new Scalar(0, 0, 255), 2);
            }
            //11-13
            if (mPrintPointArray[0][11] > 0 && mPrintPointArray[1][11] > 0
                    && mPrintPointArray[0][13] > 0 && mPrintPointArray[1][13] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][11], mPrintPointArray[1][11]),
                        new Point(mPrintPointArray[0][13], mPrintPointArray[1][13]), new Scalar(0, 0, 255), 2);
            }
            //13-15
            if (mPrintPointArray[0][13] > 0 && mPrintPointArray[1][13] > 0
                    && mPrintPointArray[0][15] > 0 && mPrintPointArray[1][15] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][13], mPrintPointArray[1][13]),
                        new Point(mPrintPointArray[0][15], mPrintPointArray[1][15]), new Scalar(0, 0, 255), 2);
            }
            //12-14
            if (mPrintPointArray[0][12] > 0 && mPrintPointArray[1][12] > 0
                    && mPrintPointArray[0][14] > 0 && mPrintPointArray[1][14] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][12], mPrintPointArray[1][12]),
                        new Point(mPrintPointArray[0][14], mPrintPointArray[1][14]), new Scalar(0, 0, 255), 2);
            }
            //14-16
            if (mPrintPointArray[0][14] > 0 && mPrintPointArray[1][14] > 0
                    && mPrintPointArray[0][16] > 0 && mPrintPointArray[1][16] > 0) {
                Imgproc.line(res_mat, new Point(mPrintPointArray[0][14], mPrintPointArray[1][14]),
                        new Point(mPrintPointArray[0][16], mPrintPointArray[1][16]), new Scalar(0, 0, 255), 2);
            }
            Imgcodecs.imwrite(im_path, res_mat);

        }
        // gb add read mp4 end

    }

    private float get(int x, int y, float[] arr) {
        if (x < 0 || y < 0 || x >= getOutputSizeX() || y >= getOutputSizeY())
            return -1;
        return arr[y * getOutputSizeX() + x];

        /*runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });*/
    }
}
