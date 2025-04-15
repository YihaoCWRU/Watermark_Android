package com.team.project

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.graphics.Color
import android.util.Log
import androidx.xr.runtime.math.clamp
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.roundToInt

class WatermarkProcessor(private val context: Context) {

    private var moduleEnc: Module? = null
    private var moduleDec: Module? = null

    // Maximum watermark string length (for detector output decoding)
    private val WATERMARK_MAX_LEN = 32

    init {
        try {
            // Load encoder and detector models (converted to .pt)
            moduleEnc = Module.load(assetFilePath("encoder.pt"))
            moduleDec = Module.load(assetFilePath("detector.pt"))
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Failed to load models", e)
        }
    }

// WatermarkProcessor.kt


    fun encryptImage(bitmap: Bitmap): Bitmap? {
        try {
            // Convert Bitmap to tensor, normalized to [0, 1]
            val inputTensor = bitmapToFloatTensor(bitmap)
            // Get residual from encoder with single input (DeepRaft encoder)
            val residualTensor = moduleEnc?.forward(IValue.from(inputTensor))?.toTensor() ?: return null
            // Generate watermarked image by adding residual to cover image and then clamping to [0, 1]
            val watermarkedTensor = (inputTensor + residualTensor).clamp(0f, 1f)
            return tensorToBitmap(watermarkedTensor)
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Encryption failed", e)
            return null
        }
    }


    operator fun Tensor.plus(other: Tensor): Tensor {
        val data1 = this.dataAsFloatArray
        val data2 = other.dataAsFloatArray
        if (data1.size != data2.size) {
            throw IllegalArgumentException("Tensors have different sizes: ${data1.size} vs ${data2.size}")
        }
        val result = FloatArray(data1.size)
        for (i in data1.indices) {
            result[i] = data1[i] + data2[i]
        }
        return Tensor.fromBlob(result, this.shape())
    }


    // Decrypt image: input is the watermarked image, outputs decoded watermark message as a string
    /**
     * Decrypts the watermarked image using the detector model.
     * Since the detector outputs a 2-element vector (for binary classification),
     * this method interprets the output as probabilities, and returns a corresponding message.
     */
    fun decryptImage(watermarkedBitmap: Bitmap): String {
        try {
            val inputTensor = bitmapToFloatTensor(watermarkedBitmap)
            // Call the detector model; it expects a watermarked image tensor as input.
            val outputTensor = moduleDec?.forward(IValue.from(inputTensor))?.toTensor() ?: return "Decryption failed"
            // Retrieve the output as a FloatArray. Expected shape is [1,2].
            val outputArray = outputTensor.dataAsFloatArray
            // Check the length to ensure it contains two values.
            if (outputArray.size < 2) {
                return "Invalid output"
            }
            // Interpret the two values as probabilities.
            // For example, if outputArray[1] > outputArray[0], we can say that the watermark is detected.
            return if (outputArray[1] > outputArray[0]) "Watermark Detected" else "Watermark Not Detected"
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Decryption failed", e)
            return "Decryption error"
        }
    }


    // Convert a Bitmap to a Tensor (normalized to [0, 1])
    private fun bitmapToFloatTensor(bitmap: Bitmap): Tensor {
        // Use TorchVision helper; here, no mean/std normalization, just scaling to [0,1]
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f))
    }

    // Convert a Tensor to a Bitmap (assumes tensor shape is [1, 3, 400, 400])
    private fun tensorToBitmap(tensor: Tensor): Bitmap {
        val scores = tensor.dataAsFloatArray
        val width = 400
        val height = 400
        val bmp = Bitmap.createBitmap(width, height, Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val channelSize = width * height
        for (i in 0 until channelSize) {
            val r = (scores[i] * 255).roundToInt().coerceIn(0, 255)
            val g = (scores[i + channelSize] * 255).roundToInt().coerceIn(0, 255)
            val b = (scores[i + 2 * channelSize] * 255).roundToInt().coerceIn(0, 255)
            pixels[i] = Color.argb(255, r, g, b)
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }

    // Convert the detector's output tensor (assumed shape [1, WATERMARK_MAX_LEN]) into a string
    private fun tensorToString(tensor: Tensor): String {
        val data = tensor.dataAsFloatArray
        val sb = StringBuilder()
        for (i in 0 until WATERMARK_MAX_LEN) {
            val charCode = (data[i] * 255).roundToInt()
            if (charCode != 0) {
                sb.append(charCode.toChar())
            }
        }
        return sb.toString()
    }

    // Copy an asset file to a local file path for Module.load usage
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        } catch (e: IOException) {
            throw RuntimeException("Error processing asset $assetName to file path", e)
        }
        return file.absolutePath
    }

    // Save a Bitmap to local storage (simple implementation: save to app private storage)
    fun saveBitmapToGallery(context: Context, bitmap: Bitmap, fileName: String) {
        try {
            val file = File(context.filesDir, fileName)
            FileOutputStream(file).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                fos.flush()
            }
            Log.i("WatermarkProcessor", "Image saved to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Failed to save image", e)
        }
    }

    fun Tensor.clamp(min: Float, max: Float): Tensor {
        val data = this.dataAsFloatArray.map { it.coerceIn(min, max) }.toFloatArray()
        return Tensor.fromBlob(data, this.shape())
    }
    /**
     * Returns the residual image computed by the encoder.
     * Optionally, a scaling factor can be applied for visualization.
     */
    fun getResidualImage(bitmap: Bitmap, scaleFactor: Float = 5f): Bitmap? {
        try {
            // Convert the input Bitmap (cover image) to tensor, normalized to [0,1]
            val inputTensor = bitmapToFloatTensor(bitmap)
            // Get the residual from the encoder; encoder.forward expects a single image tensor.
            val residualTensor = moduleEnc?.forward(IValue.from(inputTensor))?.toTensor() ?: return null
            // Multiply residual by a scaling factor for visualization (since residual's values are small)
            val scaledResidualTensor = multiplyTensor(residualTensor, scaleFactor)
            // Optional: Clamp the scaled residual to [0,1] (if necessary for visualization)
            val clippedResidualTensor = scaledResidualTensor.clamp(0f, 1f)
            // Convert the tensor to Bitmap and return it
            return tensorToBitmap(clippedResidualTensor)
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Get residual image failed", e)
            return null
        }
    }

    // Extension function to multiply a Tensor element-wise by a scalar
    fun multiplyTensor(tensor: Tensor, factor: Float): Tensor {
        val data = tensor.dataAsFloatArray.map { it * factor }.toFloatArray()
        return Tensor.fromBlob(data, tensor.shape())
    }

}
