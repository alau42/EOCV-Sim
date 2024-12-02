/*
 * Copyright (c) 2023 Sebastian Erives
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package org.firstinspires.ftc.teamcode.processor;

import android.graphics.Canvas;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

public class SampleDetectionProcessorYCrCb implements VisionProcessor {

    Mat clrMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedMat = new Mat();

    int redThreshold = 200;
    int yellowThreshold = 80;
    int blueThreshold = 150;

    int crChannelIndex = 1;
    int cbChannelIndex = 2;

    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));

    static Scalar green = new Scalar(0, 255, 0);

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
    }

    @Override
    public Object processFrame(Mat input, long captureTimeNanos) {

        for(MatOfPoint contour : findContours(input, "RED"))
        {
            analyzeContour(contour, input);
        }

        return null;
    }

    private void analyzeContour(MatOfPoint contour, Mat input) {

        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        RotatedRect rotatedRectFitToContour = Imgproc.minAreaRect(contour2f);
        drawRotatedRect(rotatedRectFitToContour, input);

        double rotRectAngle = rotatedRectFitToContour.angle;
        if (rotatedRectFitToContour.size.width < rotatedRectFitToContour.size.height)
        {
            rotRectAngle += 90;
        }

        double angle = 180-(rotRectAngle);

        drawTagText(rotatedRectFitToContour, Integer.toString((int) Math.round(angle))+" deg", input);

    }

    ArrayList<MatOfPoint> findContours(Mat input, String color)
    {
        int threshold = 0;
        int channel = color.equals("RED") ? crChannelIndex : cbChannelIndex;
        int threshMode  = color.equals("YELLOW") ? Imgproc.THRESH_BINARY_INV : Imgproc.THRESH_BINARY;
        if(color.equals("RED")) threshold = redThreshold;
        if(color.equals("YELLOW")) threshold = yellowThreshold;
        if(color.equals("BLUE")) threshold = blueThreshold;

        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.cvtColor(input, clrMat, Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(clrMat, clrMat, channel);
        clrMat.copyTo(input);
        Imgproc.threshold(clrMat, thresholdMat, threshold, 255, threshMode);
        morphMask(thresholdMat, morphedMat);
        Imgproc.findContours(morphedMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        return contoursList;
    }

    void morphMask(Mat input, Mat output)
    {
        Imgproc.erode(input, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);

        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);
    }


    static void drawRotatedRect(RotatedRect rect, Mat drawOn)
    {
        Point[] points = new Point[4];
        rect.points(points);

        for(int i = 0; i < 4; ++i)
        {
            Imgproc.line(drawOn, points[i], points[(i+1)%4], green, 2);
        }
    }

    static void drawTagText(RotatedRect rect, String text, Mat mat)
    {
        Imgproc.putText(
                mat, // The buffer we're drawing on
                text, // The text we're drawing
                new Point( // The anchor point for the text
                        rect.center.x-27.5,  // x anchor point
                        rect.center.y+5), // y anchor point
                Imgproc.FONT_HERSHEY_PLAIN, // Font
                1, // Font size
                green, // Font color
                2); // Font thickness
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
    }


}

