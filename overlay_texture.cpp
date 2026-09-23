#include "overlay_internal.h"

bool DXOverlay::CreateShaders() {
    HRESULT hr = S_OK;
    ComPtr<ID3DBlob> vertexBlob, pixelBlob, errorBlob;

    // 编译顶点着色器
    hr = D3DCompile(
        g_vertexShaderCode,
        strlen(g_vertexShaderCode),
        nullptr,
        nullptr,
        nullptr,
        "VS",
        "vs_5_0",
        0,
        0,
        &vertexBlob,
        &errorBlob
    );

    if (FAILED(hr)) {
        if (errorBlob) {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        }
        char msg[256];
        sprintf_s(msg, "Failed to compile vertex shader, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        return false;
    }

    hr = m_device->CreateVertexShader(
        vertexBlob->GetBufferPointer(),
        vertexBlob->GetBufferSize(),
        nullptr,
        &m_vertexShader
    );

    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "Failed to create vertex shader, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        return false;
    }

    // 创建输入布局
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    hr = m_device->CreateInputLayout(
        layout,
        ARRAYSIZE(layout),
        vertexBlob->GetBufferPointer(),
        vertexBlob->GetBufferSize(),
        &m_inputLayout
    );

    if (FAILED(hr)) return false;

    // 编译像素着色器
    hr = D3DCompile(
        g_pixelShaderCode,
        strlen(g_pixelShaderCode),
        nullptr,
        nullptr,
        nullptr,
        "PS",
        "ps_5_0",
        0,
        0,
        &pixelBlob,
        &errorBlob
    );

    if (FAILED(hr)) {
        if (errorBlob) {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        }
        char msg[256];
        sprintf_s(msg, "Failed to compile pixel shader, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        return false;
    }

    hr = m_device->CreatePixelShader(
        pixelBlob->GetBufferPointer(),
        pixelBlob->GetBufferSize(),
        nullptr,
        &m_pixelShader
    );

    if (FAILED(hr)) return false;

    // 创建顶点缓冲区（全屏 Quad）
    Vertex vertices[] = {
        { -1.0f,  1.0f, 0.0f, 0.0f }, // 左上
        {  1.0f,  1.0f, 1.0f, 0.0f }, // 右上
        { -1.0f, -1.0f, 0.0f, 1.0f }, // 左下
        {  1.0f, -1.0f, 1.0f, 1.0f }, // 右下
    };

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(vertices);
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = vertices;

    hr = m_device->CreateBuffer(&bufferDesc, &initData, &m_vertexBuffer);
    if (FAILED(hr)) return false;

    // 创建常量缓冲区
    bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.ByteWidth = sizeof(ConstantBufferData);
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    hr = m_device->CreateBuffer(&bufferDesc, nullptr, &m_constantBuffer);
    if (FAILED(hr)) return false;

    // 动态模板使用 point sampling，避免纯色块边缘被线性插值出中间色。
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = m_device->CreateSamplerState(&samplerDesc, &m_pointSamplerState);
    if (FAILED(hr)) return false;

    // 创建混合状态（预乘 alpha 混合）
    // Shader 输出已经是预乘的，所以 SrcBlend = ONE
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = TRUE;  // 启用混合
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;  // 预乘 alpha: 源已经乘过 alpha
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;  // 1 - srcAlpha
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    hr = m_device->CreateBlendState(&blendDesc, &m_blendState);
    if (FAILED(hr)) return false;

    return true;
}

bool DXOverlay::LoadTextures(const std::vector<std::wstring>& imagePaths) {
    if (imagePaths.empty()) {
        LogToFile("No image paths provided\n");
        return false;
    }

    // 缓存图像路径，用于设备恢复时重新加载
    m_imagePaths = imagePaths;

    m_textureCount = static_cast<int>(imagePaths.size());

    char countMsg[256];
    sprintf_s(countMsg, "Starting texture loading: %d images\n", m_textureCount);
    LogToFile(countMsg);
    m_dynamicTexturesUseSourceAlpha = false;
    bool sawChannelEncodedTexture = false;
    bool sawLegacyTexture = false;

    // 第一步：加载所有图片到临时纹理，获取尺寸
    struct TempTexture {
        ComPtr<ID3D11Texture2D> texture;
        int width;
        int height;
    };
    std::vector<TempTexture> tempTextures(imagePaths.size());

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        ComPtr<ID3D11ShaderResourceView> tempSRV;  // 临时 SRV，后面不需要
        bool hasNonOpaqueAlpha = false;
        bool isChannelEncoded = false;
        if (!LoadTextureFromFile(
                imagePaths[i],
                tempSRV,
                tempTextures[i].width,
                tempTextures[i].height,
                &hasNonOpaqueAlpha,
                &isChannelEncoded)) {
            char msg[512];
            sprintf_s(msg, "Failed to load texture %zu\n", i);
            LogToFile(msg);
            return false;
        }
        m_dynamicTexturesUseSourceAlpha = m_dynamicTexturesUseSourceAlpha || hasNonOpaqueAlpha;
        sawChannelEncodedTexture = sawChannelEncodedTexture || isChannelEncoded;
        sawLegacyTexture = sawLegacyTexture || !isChannelEncoded;

        // 从 SRV 提取 Texture2D 资源
        ID3D11Resource* resource = nullptr;
        tempSRV->GetResource(&resource);
        if (resource) {
            HRESULT hrQuery = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tempTextures[i].texture);
            resource->Release();
            if (FAILED(hrQuery) || !tempTextures[i].texture) {
                char errMsg[256];
                sprintf_s(errMsg, "Failed to extract Texture2D from SRV for texture %zu, HRESULT: 0x%08X\n", i, hrQuery);
                LogToFile(errMsg);
                return false;
            }
        } else {
            char errMsg[256];
            sprintf_s(errMsg, "Failed to get resource from SRV for texture %zu\n", i);
            LogToFile(errMsg);
            return false;
        }

        char loadMsg[256];
        sprintf_s(loadMsg, "Loaded texture %zu: %dx%d\n", i, tempTextures[i].width, tempTextures[i].height);
        LogToFile(loadMsg);

        if (i == 0) {
            m_imageWidth = tempTextures[i].width;
            m_imageHeight = tempTextures[i].height;

            // 设置窗口为全屏（覆盖整个目标显示器）
            char posMsg[256];
            sprintf_s(posMsg, "Setting window to fullscreen: %dx%d at (%d, %d)\n",
                m_screenWidth, m_screenHeight, m_windowX, m_windowY);
            LogToFile(posMsg);

            m_ignoreResizeEvents.store(true);
            SetWindowPos(m_hwnd, HWND_TOPMOST, m_windowX, m_windowY,
                m_screenWidth, m_screenHeight, SWP_NOACTIVATE | SWP_NOSENDCHANGING);
            m_ignoreResizeEvents.store(false);

            if (!ResizeSwapChainBuffers(m_screenWidth, m_screenHeight, "initial texture load")) {
                return false;
            }
        }
    }

    if (sawChannelEncodedTexture && sawLegacyTexture) {
        LogToFile("ERROR: Mixed channel-encoded and legacy templates are not supported\n");
        return false;
    }
    m_channelEncodedTemplates = sawChannelEncodedTexture;
    LogToFile(m_channelEncodedTemplates
        ? "Template format: independent Y/Cr/Cb channel data (Cr dynamic, Cb static)\n"
        : "Template format: legacy combined RGB(A); independent channel strength unavailable\n");
    if (!m_channelEncodedTemplates &&
        std::fabs(m_staticAlpha - m_dynamicAlpha) > 0.000001f) {
        m_lastError =
            "Independent static Cb and dynamic Cr strengths require v4.2 channel-data "
            "templates. Set watermark_id to regenerate the templates, or use equal strengths "
            "with legacy RGB(A) images.";
        LogToFile("ERROR: Different Cb/Cr strengths requested with a legacy combined template\n");
        return false;
    }

    LogToFile("All temp textures loaded, starting texture array creation...\n");

    // 获取第一个纹理的格式，确保数组格式与源纹理一致
    D3D11_TEXTURE2D_DESC firstTexDesc;
    tempTextures[0].texture->GetDesc(&firstTexDesc);

    char formatMsg[256];
    sprintf_s(formatMsg, "Source texture format: %d, creating array with same format\n", firstTexDesc.Format);
    LogToFile(formatMsg);

        // 【稳定性检查】验证所有纹理的尺寸和格式是否一致
    for (size_t i = 1; i < tempTextures.size(); ++i) {
        D3D11_TEXTURE2D_DESC desc;
        tempTextures[i].texture->GetDesc(&desc);

        if (desc.Width != firstTexDesc.Width || desc.Height != firstTexDesc.Height) {
            char errMsg[512];
            sprintf_s(errMsg, "ERROR: Texture %zu size (%dx%d) does not match first texture (%dx%d)\n"
                              "All images must have the same resolution for Texture Array.\n",
                      i, desc.Width, desc.Height, firstTexDesc.Width, firstTexDesc.Height);
            LogToFile(errMsg);
            return false;
        }

        if (desc.Format != firstTexDesc.Format) {
            char errMsg[512];
            sprintf_s(errMsg, "ERROR: Texture %zu format (%d) does not match first texture (%d)\n"
                              "All images must have the same pixel format.\n",
                      i, desc.Format, firstTexDesc.Format);
            LogToFile(errMsg);
            return false;
        }
    }

    // 第二步：创建纹理数组（Texture2DArray）- 使用与源纹理相同的格式
    D3D11_TEXTURE2D_DESC arrayDesc = {};
    arrayDesc.Width = m_imageWidth;
    arrayDesc.Height = m_imageHeight;
    arrayDesc.MipLevels = 1;
    arrayDesc.ArraySize = m_textureCount;  // 数组大小 = 纹理数量
    arrayDesc.Format = firstTexDesc.Format;  // 使用源纹理的格式！
    arrayDesc.SampleDesc.Count = 1;
    arrayDesc.SampleDesc.Quality = 0;
    arrayDesc.Usage = D3D11_USAGE_DEFAULT;
    arrayDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    arrayDesc.CPUAccessFlags = 0;
    arrayDesc.MiscFlags = 0;

    HRESULT hr = m_device->CreateTexture2D(&arrayDesc, nullptr, &m_textureArray);
    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "Failed to create texture array, HRESULT: 0x%08X\n", hr);
        LogToFile(msg);
        return false;
    }

    // 第三步：将每个纹理复制到纹理数组的相应层
    LogToFile("Copying textures to texture array...\n");
    for (int i = 0; i < m_textureCount; ++i) {
        if (!tempTextures[i].texture) {
            char errMsg[256];
            sprintf_s(errMsg, "ERROR: tempTextures[%d].texture is NULL!\n", i);
            LogToFile(errMsg);
            return false;
        }

        // 获取源纹理描述
        D3D11_TEXTURE2D_DESC srcDesc;
        tempTextures[i].texture->GetDesc(&srcDesc);

        char srcInfo[512];
        sprintf_s(srcInfo, "Source texture %d: %dx%d, Format=%d, Usage=%d, BindFlags=0x%X\n",
                 i, srcDesc.Width, srcDesc.Height, srcDesc.Format, srcDesc.Usage, srcDesc.BindFlags);
        LogToFile(srcInfo);

        UINT subresourceIndex = D3D11CalcSubresource(0, i, 1);  // mip=0, array_slice=i

        char copyMsg[256];
        sprintf_s(copyMsg, "Copying texture %d to array slice %d (subresource=%u)\n", i, i, subresourceIndex);
        LogToFile(copyMsg);

        m_context->CopySubresourceRegion(
            m_textureArray.Get(),        // 目标：纹理数组
            subresourceIndex,            // 目标子资源索引
            0, 0, 0,                     // 目标位置 (x, y, z)
            tempTextures[i].texture.Get(), // 源：单个纹理
            0,                           // 源子资源索引
            nullptr                      // 复制整个资源
        );
    }
    LogToFile("All textures copied to array\n");

    // 第四步：为纹理数组创建 Shader Resource View
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = arrayDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    srvDesc.Texture2DArray.MostDetailedMip = 0;
    srvDesc.Texture2DArray.MipLevels = 1;
    srvDesc.Texture2DArray.FirstArraySlice = 0;
    srvDesc.Texture2DArray.ArraySize = m_textureCount;

    hr = m_device->CreateShaderResourceView(m_textureArray.Get(), &srvDesc, &m_textureArraySRV);
    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "Failed to create texture array SRV, HRESULT: 0x%08X\n", hr);
        LogToFile(msg);
        return false;
    }

    char successMsg[256];
    sprintf_s(successMsg, "Texture array created successfully: %d textures, %dx%d\n",
              m_textureCount, m_imageWidth, m_imageHeight);
    LogToFile(successMsg);

    // 【一次性绑定】将纹理数组绑定到像素着色器。
    ID3D11ShaderResourceView* srv = m_textureArraySRV.Get();
    m_context->PSSetShaderResources(0, 1, &srv);

    char bindMsg[512];
    sprintf_s(bindMsg, "Texture array bound to pixel shader (one-time binding)\n  SRV pointer: %p\n  Array size: %d\n  Resolution: %dx%d\n",
             srv, m_textureCount, m_imageWidth, m_imageHeight);
    LogToFile(bindMsg);

    return true;
}

bool DXOverlay::LoadTextureFromFile(
    const std::wstring& path,
    ComPtr<ID3D11ShaderResourceView>& srv,
    int& width,
    int& height,
    bool* hasNonOpaqueAlpha,
    bool* isChannelEncoded) {
    // 检查文件是否存在
    DWORD fileAttr = GetFileAttributesW(path.c_str());
    if (fileAttr == INVALID_FILE_ATTRIBUTES || (fileAttr & FILE_ATTRIBUTE_DIRECTORY)) {
        char msg[512];
        sprintf_s(msg, "File does not exist or is a directory: %ws\n", path.c_str());
        OutputDebugStringA(msg);
        return false;
    }

    // 使用 WIC 加载图像
    ComPtr<IWICImagingFactory> wicFactory;
    HRESULT hr = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    );

    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "Failed to create WIC factory, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        return false;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    hr = wicFactory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnDemand,
        &decoder
    );

    if (FAILED(hr)) {
        // 使用 WideCharToMultiByte 进行安全的字符串转换
        int pathLen = WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (pathLen > 0) {
            std::vector<char> pathA(pathLen);
            WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, pathA.data(), pathLen, nullptr, nullptr);
            char msg[1024];
            sprintf_s(msg, "Failed to create decoder for file: %s, HRESULT: 0x%08X\n", pathA.data(), hr);
            OutputDebugStringA(msg);
        }

        // 尝试使用完整路径
        wchar_t fullPath[MAX_PATH];
        DWORD result = GetFullPathNameW(path.c_str(), MAX_PATH, fullPath, nullptr);
        if (result > 0 && result < MAX_PATH) {
            int fullPathLen = WideCharToMultiByte(CP_UTF8, 0, fullPath, -1, nullptr, 0, nullptr, nullptr);
            if (fullPathLen > 0) {
                std::vector<char> fullPathA(fullPathLen);
                WideCharToMultiByte(CP_UTF8, 0, fullPath, -1, fullPathA.data(), fullPathLen, nullptr, nullptr);
                char msg[1024];
                sprintf_s(msg, "Full path: %s\n", fullPathA.data());
                OutputDebugStringA(msg);

                // 检查文件是否存在
                DWORD attr = GetFileAttributesW(fullPath);
                if (attr != INVALID_FILE_ATTRIBUTES) {
                    sprintf_s(msg, "File exists with attributes: 0x%08X\n", attr);
                    OutputDebugStringA(msg);
                } else {
                    sprintf_s(msg, "File does not exist, error: %lu\n", ::GetLastError());
                    OutputDebugStringA(msg);
                }
            }
        }
        return false;
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) return false;

    hr = frame->GetSize((UINT*)&width, (UINT*)&height);
    if (FAILED(hr)) return false;

    ComPtr<IWICFormatConverter> converter;
    hr = wicFactory->CreateFormatConverter(&converter);
    if (FAILED(hr)) return false;

    hr = converter->Initialize(
        frame.Get(),
        GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom
    );

    if (FAILED(hr)) return false;

    // 读取像素数据
    UINT stride = width * 4;
    UINT bufferSize = stride * height;
    std::vector<BYTE> buffer(bufferSize);

    hr = converter->CopyPixels(nullptr, stride, bufferSize, buffer.data());
    if (FAILED(hr)) return false;

    bool anyNonOpaqueAlpha = false;
    bool allChannelMarkers = bufferSize >= 4;
    for (UINT offset = 0; offset + 3 < bufferSize; offset += 4) {
        const BYTE encodedY = buffer[offset];  // logical PNG R channel
        const BYTE alpha = buffer[offset + 3];
        anyNonOpaqueAlpha = anyNonOpaqueAlpha || alpha != 255;
        allChannelMarkers = allChannelMarkers && encodedY == 128 && alpha == 254;
    }
    if (isChannelEncoded) {
        *isChannelEncoded = allChannelMarkers;
    }
    if (hasNonOpaqueAlpha) {
        // Alpha=254 on every pixel is a format marker, not an opacity mask.
        *hasNonOpaqueAlpha = anyNonOpaqueAlpha && !allChannelMarkers;
    }

    // 创建纹理
    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = buffer.data();
    initData.SysMemPitch = stride;

    ComPtr<ID3D11Texture2D> texture;
    hr = m_device->CreateTexture2D(&texDesc, &initData, &texture);
    if (FAILED(hr)) return false;

    // 创建着色器资源视图
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;

    hr = m_device->CreateShaderResourceView(texture.Get(), &srvDesc, &srv);
    if (FAILED(hr)) return false;

    return true;
}
