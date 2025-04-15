package com.team.project

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var selectImageButton: Button
    private lateinit var encryptButton: Button
    // 如果需要单独触发解码测试，则保留解码按钮
    private lateinit var decryptButton: Button
    private lateinit var originalImageView: ImageView
    private lateinit var processedImageView: ImageView
    // 用于显示最终解码出的水印信息
    private lateinit var watermarkDisplayTextView: TextView

    private var selectedBitmap: Bitmap? = null
    private var processedBitmap: Bitmap? = null

    private lateinit var watermarkProcessor: WatermarkProcessor

    // 使用 ActivityResultContracts 选择图片
    private val selectImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bmp = uriToBitmap(it)
                // 将图片中心裁剪并缩放到 400×400
                selectedBitmap = cropToSquareAndResize(bmp, 400)
                originalImageView.setImageBitmap(selectedBitmap)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化 WatermarkProcessor（内部加载 encoder.pt 和 detector.pt 模型）
        watermarkProcessor = WatermarkProcessor(this)

        selectImageButton = findViewById(R.id.btn_select_image)
        encryptButton = findViewById(R.id.btn_encrypt)
        decryptButton = findViewById(R.id.btn_decrypt)
        originalImageView = findViewById(R.id.iv_original)
        processedImageView = findViewById(R.id.iv_processed)
        watermarkDisplayTextView = findViewById(R.id.tv_decode_result)

        selectImageButton.setOnClickListener {
            // 调用系统图片选择器
            selectImageLauncher.launch("image/*")
        }

        encryptButton.setOnClickListener {
            val bitmap = selectedBitmap
            if (bitmap != null) {
                // 使用 Encoder 模型进行加密，注意：Encoder.forward 只需要传入图像张量
                processedBitmap = watermarkProcessor.encryptImage(bitmap)
                processedBitmap?.let { watermarked ->
                    // 在 ImageView 中显示水印图像
                    processedImageView.setImageBitmap(watermarked)
                    // 立即使用 Detector 模型提取嵌入的水印信息
                    val extractedWatermark = watermarkProcessor.decryptImage(watermarked)
                    // 在 TextView 中显示提取出来的水印信息
                    watermarkDisplayTextView.text = "Embedded watermark: $extractedWatermark"
                    // 保存水印图像（可选）
                    watermarkProcessor.saveBitmapToGallery(this, watermarked, "watermarked_${System.currentTimeMillis()}.png")
                }
            }
        }

        // 如果你希望单独触发解码，则使用解码按钮
        decryptButton.setOnClickListener {
            val bitmap = processedBitmap
            if (bitmap != null) {
                val extractedWatermark = watermarkProcessor.decryptImage(bitmap)
                watermarkDisplayTextView.text = "Extracted watermark: $extractedWatermark"
            }
        }
    }

    // 将 URI 转换为 Bitmap
    private fun uriToBitmap(uri: Uri): Bitmap {
        val resolver: ContentResolver = contentResolver
        val inputStream: InputStream? = resolver.openInputStream(uri)
        return BitmapFactory.decodeStream(inputStream)
    }

    // 中心裁剪并缩放图片到指定尺寸
    private fun cropToSquareAndResize(srcBitmap: Bitmap, size: Int): Bitmap {
        val width = srcBitmap.width
        val height = srcBitmap.height
        val newEdge = if (width < height) width else height
        val xOffset = (width - newEdge) / 2
        val yOffset = (height - newEdge) / 2
        val cropped = Bitmap.createBitmap(srcBitmap, xOffset, yOffset, newEdge, newEdge)
        return Bitmap.createScaledBitmap(cropped, size, size, true)
    }
}
