package com.team.project

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.graphics.Color
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.roundToInt

class temp(private val context: Context) {

    private var moduleEnc: Module? = null
    private var moduleDec: Module? = null

    // Maximum length of the watermark string (e.g., 32 characters)
    private val WATERMARK_MAX_LEN = 32

    init {
        try {
            // Load encryption model
            moduleEnc = Module.load(assetFilePath("vine_r_enc.pt"))
            // Load decryption model
            moduleDec = Module.load(assetFilePath("vine_r_dec.pt"))
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Failed to load models", e)
        }
    }

    // Encrypt: embed the custom watermark string into the image
    fun encryptImage(bitmap: Bitmap, watermarkText: String): Bitmap? {
        try {
            // Convert the image to a tensor, normalized to [0,1]
            val inputTensor = bitmapToFloatTensor(bitmap)

            // Convert the watermark string into a fixed-length tensor representation
            val watermarkTensor = stringToTensor(watermarkText)

            // Call forward() by providing two separate arguments
            val outputTensor = moduleEnc?.forward(
                IValue.from(inputTensor),
                IValue.from(watermarkTensor)
            )?.toTensor() ?: return null

            // Convert the output tensor back to a Bitmap (assuming shape [1,3,512,512])
            return tensorToBitmap(outputTensor)
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Encryption failed", e)
            return null
        }
    }


    // Decrypt: extract the watermark information from the watermarked image
    fun decryptImage(bitmap: Bitmap): String {
        try {
            val inputTensor = bitmapToFloatTensor(bitmap)
            // Assume the decryption model only accepts an image tensor
            val outputTensor = moduleDec?.forward(IValue.from(inputTensor))?.toTensor() ?: return "Decryption failed"
            // Parse the output tensor into a string
            return tensorToString(outputTensor)
        } catch (e: Exception) {
            Log.e("WatermarkProcessor", "Decryption failed", e)
            return "Decryption error"
        }
    }

    // Convert a Bitmap to a tensor normalized to [0,1]
    private fun bitmapToFloatTensor(bitmap: Bitmap): Tensor {
        // Use TorchVision helper; here we do not apply mean/STD normalization, keeping the original normalization
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f))
    }

    // Convert the output tensor back to a Bitmap
    private fun tensorToBitmap(tensor: Tensor): Bitmap {
        Log.d("WatermarkProcessor", "Tensor shape: " + tensor.shape().contentToString())
// 打印前几个数值
        Log.d("WatermarkProcessor", "First 10 tensor values: " + tensor.dataAsFloatArray.take(10).toString())

        // Retrieve tensor data, assuming shape is [1,3,512,512]
        val scores = tensor.dataAsFloatArray
        val width = 400
        val height = 400
        val bmp = Bitmap.createBitmap(width, height, Config.ARGB_8888)



        val pixels = IntArray(width * height)
        // Assume tensor data ordering is [channel, height, width]
        val channelSize = width * height



        for (i in 0 until channelSize) {
            // Get RGB values and denormalize to [0,255]
            val r = (scores[i] * 255).roundToInt().coerceIn(0, 255)
            val g = (scores[i + channelSize] * 255).roundToInt().coerceIn(0, 255)
            val b = (scores[i + 2 * channelSize] * 255).roundToInt().coerceIn(0, 255)
            pixels[i] = Color.argb(255, r, g, b)
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }

    // Convert the watermark string to a tensor.
    // Conversion rule: for each character, take its ASCII code divided by 255.
    // The tensor shape is [1, WATERMARK_MAX_LEN].
    private fun stringToTensor(text: String): Tensor {
        val arr = FloatArray(WATERMARK_MAX_LEN) { 0f }
        val len = text.length.coerceAtMost(WATERMARK_MAX_LEN)
        for (i in 0 until len) {
            arr[i] = (text[i].code / 255f)
        }
        return Tensor.fromBlob(arr, longArrayOf(1, WATERMARK_MAX_LEN.toLong()))
    }

    // Convert the decryption model's output tensor back into a string
    private fun tensorToString(tensor: Tensor): String {
        val data = tensor.dataAsFloatArray
        val sb = StringBuilder()
        // Assume output tensor shape is [1, WATERMARK_MAX_LEN]
        for (i in 0 until WATERMARK_MAX_LEN) {
            // Denormalize to ASCII range and convert to character (rounded)
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
            throw RuntimeException("Error processing asset $assetName to file path")
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


}
