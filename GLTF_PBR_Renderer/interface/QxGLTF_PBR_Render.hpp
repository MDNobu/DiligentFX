#pragma once

#include <unordered_map>
#include <functional>
#include <mutex>
#include <vector>

#include "../../../DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "../../../DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "../../../DiligentCore/Common/interface/HashUtils.hpp"
#include "../../../DiligentTools/AssetLoader/interface/GLTFLoader.hpp"

namespace Diligent
{

#include "Shaders/GLTF_PBR/public/QxGLTF_PBR_Structures.hlsl"

class QxGLTF_PBR_Render
{
public:
    // render create info
    struct CreateInfo
    {
        // render target format
        TEXTURE_FORMAT RTVFmt = TEX_FORMAT_UNKNOWN;

        // depth buffer format

        // 注意：如果rtvfmt和dsvfmt都是Unkown则不会创建pso等资源，期望applicaiton用自定义render callback
        TEXTURE_FORMAT DSVFmt = TEX_FORMAT_UNKNOWN;

        // 表示 正面 是不是逆时针
        bool FrontCCW = false;

        // 是否允许使用debug view
        bool AllowDebugView = false;

        // 是否使用ibl
        bool UseIBL = false;

        // 是否用 ao texture
        bool UseAO  = true;

        // 是否用emissive texture
        bool UseEmissive = true;

        // true时，pipeline state会和immutable samplers一起编译
        // false时，从texture view取sampler
        // #TODO 这个设计在UE4中也有，但DX的管线没查到，什么用???
        bool UseImmutableSamplers = true;

        // wheter to use texture altas
        bool UseTextureAtlas = false;

        static const SamplerDesc DefaultSampler;

        // color map sampler
        SamplerDesc ColorMapImmutableSampler =
            DefaultSampler;

        // physical description map sampler
        SamplerDesc PhysicalMapImmutableSampler
            = DefaultSampler;

        // normal map sampler
        SamplerDesc NormalMapImmutableSampler = DefaultSampler;

        // ao texture sampler
        SamplerDesc AOMapImmutableSampler = DefaultSampler;

        // Immutable sampler for emmisive map
        SamplerDesc EmissiveMapImmutableSampler = DefaultSampler;

        // Maxinum of joints
        Uint32 MaxJointCount = 64;
    };

    QxGLTF_PBR_Render(IRenderDevice* pDevice,
        IDeviceContext* pCtx,
        const CreateInfo& CI);

    // Rendering Information
    struct RenderInfo
    {
        // model transform所有gltf以这个为根节点
        float4x4 ModelTransform = float4x4::Identity();

        // Alpha Mode flags
        enum ALPHA_MODE_FLAGS : Uint32
        {
            // Rendering Nothing
            ALPHA_MODE_FLAGS_None = 0,

            ALPHA_MODE_FLAGS_OPAQUE = 1 << GLTF::Material::ALPHA_MODE_OPAQUE,

            ALPHA_MODE_FLAGS_MASK = 1 << GLTF::Material::ALPHA_MODE_MASK,

            ALPHA_MODE_FLAGS_BLEND = 1 << GLTF::Material::ALPHA_MODE_BLEND,

            // render all materials
            ALPHA_MODE_FLAGS_ALL =
                ALPHA_MODE_FLAGS_OPAQUE | ALPHA_MODE_FLAGS_MASK | ALPHA_MODE_FLAGS_BLEND
        };

        // flag indicating which alpha mode to render
        ALPHA_MODE_FLAGS AlphaModes = ALPHA_MODE_FLAGS_ALL;

        // Debug view type
        enum class DebugViewType : int
        {
            None = 0,
            BaseColor = 1,
            Transparency = 2,
            NormalMap = 3,
            Occlusion = 4,
            Emissive = 5,
            Metalic = 6,
            Roughness = 7,
            DiffuseColor = 8,
            SpecularColor = 8,
            Reflectance90 = 10,
            MeshNormal = 11,
            PerturbedNormal = 12,
            NdotV = 13,
            DiffuseIBL = 14,
            SpecularIBL = 15,
            NumDebugViews
        };

        DebugViewType DebugView = DebugViewType::None;

        float OcclusionStrength = 1;

        float EmissionScale = 1;

        float IBLScale = 1;

        // average log lumninace used by tone mapping
        float AverageLogLum = 0.3f;

        // middle gray value used by tone mapping
        float MiddleGray = 0.18f;
        
        // white point value used by tone mapping
        float WhitePoint = 1.f;
    };

    // gltf model shader resource binding information
    struct ModelResourceBindings
    {
        void Clear()
        {
            MaterialSRB.clear();
        }

        std::vector<RefCntAutoPtr<IShaderResourceBinding>> MaterialSRB;
    };

    // gltf resouce cache shader resource binding information
    struct ResourceCacheBindings
    {
        /// Resource version
        Uint32 Version = ~0u;

        RefCntAutoPtr<IShaderResourceBinding> pSRB;
    };

    
    // Render a GLTF Model
    void Render(IDeviceContext* pCtx,
        GLTF::Model& GLTFModel,
        const RenderInfo& RenderParams,
        ModelResourceBindings* pModelBindings,
        ResourceCacheBindings* pCacheBindings = nullptr);
    
    ModelResourceBindings CreateResourceBindings(
        GLTF::Model& GLTFModel,
        IBuffer* pCameraAttribs,
        IBuffer* pLightAttribs
        );
    
    
    // 预计算IBL用的cubemap
    void PreComputeCubemaps(IRenderDevice* pDevice,
        IDeviceContext* pCtx,
        ITextureView* pEnvironmentMap);

#pragma region SRVGetters
    ITextureView* GetIrradianceCubeSRV()
    {
        return m_pIrradianceCubeSRV;
    }
    ITextureView* GetPrefilteredEnvMapSRV()
    {
        return m_pPrefilteredEnvMapSRV;
    }
    
    ITextureView* GetBRDFLUTSRV()
    {
        return m_pBRDF_LUT_SRV;
    }
    
    ITextureView* GetWhiteTexSRV()
    {
        return m_pWhiteTexSRV;
    }
    ITextureView* GetBlackTexSRV()
    {
        return m_pBlackTexSRV;
    }
    ITextureView* GetDefaultNormalMapSRV()
    {
        return m_pDefaultNormalMapSRV;
    }
#pragma endregion
    
    // create a shader resource binding for given material
    void CreateMaterialSRB(GLTF::Model& Model,
        GLTF::Material& Material,
        IBuffer* pCameraAttribs,
        IBuffer* pLightAttribs,
        IPipelineState* pPSO,
        IShaderResourceBinding** ppMaterialSRB);
    
    // Create a shader resource binding for a GLTF resource cache
    void CreateResourceCacheSRB(
        IRenderDevice* pDevice,
        IDeviceContext* pCtx,
        GLTF::ResourceCacheUseInfo& CacheUseInfo,
        IBuffer* pCameraAttribs,
        IBuffer* pLightAttribs,
        IPipelineState* pPSO,
        IShaderResourceBinding** ppCacheSRB);
    
    /// Prepares the renderer for rendering objects.
    /// This method must be called at least once per frame.
    void Begin(IDeviceContext* pCtx);
    
    /// Prepares the renderer for rendering objects from the resource cache.
    /// This method must be called at least once per frame before the first object
    /// from the cache is rendered.
    void Begin(
        IRenderDevice* pDevice,
        IDeviceContext* pCtx,
        GLTF::ResourceCacheUseInfo& CacheUseInfo,
        ResourceCacheBindings& Bindings,
        IBuffer* pCameraAttribs,
        IBuffer* pLightAttribs,
        IPipelineState* pPSO = nullptr
        );
private:

    void PrecomputeBRDF(IRenderDevice* pDevice,
        IDeviceContext* pCtx);
    
    void CreatePSO(IRenderDevice* pDevice);

    void InitCommonSRBVars(
        IShaderResourceBinding* pSRB,
        IBuffer* pCameraAttribs,
        IBuffer* pLightAttribs
        );
    
    struct PSOKey
    {
        PSOKey() noexcept {  };
        PSOKey(GLTF::Material::ALPHA_MODE InAlphaMode,
            bool InDoubleSided)
                : AlphaMode(InAlphaMode), DoubleSided(InDoubleSided)
        {
        }

        bool operator==(const PSOKey& rhs) const
        {
            return AlphaMode == rhs.AlphaMode && DoubleSided == rhs.DoubleSided;
        }

        bool operator!=(const PSOKey& rhs) const
        {
            return AlphaMode != rhs.AlphaMode || DoubleSided != rhs.DoubleSided;
        }

        GLTF::Material::ALPHA_MODE AlphaMode =
            GLTF::Material::ALPHA_MODE_OPAQUE;
        bool DoubleSided = false;
    };

    static size_t GetPSOIdx(const PSOKey& Key)
    {
        size_t PSOIdx;

        PSOIdx = Key.AlphaMode == GLTF::Material::ALPHA_MODE_BLEND ? 1 : 0;
        PSOIdx = PSOIdx * 2 + (Key.DoubleSided ? 1 : 0);
        return PSOIdx;
    }

    void AddPSO(const PSOKey& Key, RefCntAutoPtr<IPipelineState> pPSO)
    {
        size_t Idx = GetPSOIdx(Key);
        if (Idx >= m_PSOCaches.size())
        {
            m_PSOCaches.resize(Idx + 1);
        }
        VERIFY_EXPR(!m_PSOCaches[Idx]);
        m_PSOCaches[Idx] = std::move(pPSO);
    }

    IPipelineState* GetPSO(const PSOKey& Key)
    {
        size_t Idx = GetPSOIdx(Key);
        VERIFY_EXPR(Idx < m_PSOCaches.size());
        IPipelineState* res = nullptr;
        if (Idx < m_PSOCaches.size())
        {
            res = m_PSOCaches[Idx].RawPtr();
        }
        // return Idx < m_PSOCaches.size() ?  m_PSOCaches[Idx] : nullptr;
        return res;
    }

    const CreateInfo m_Settings;
    
    static constexpr Uint32 BRDF_LUT_Dim = 512;
    RefCntAutoPtr<ITextureView> m_pBRDF_LUT_SRV;
    
    std::vector<RefCntAutoPtr<IPipelineState>> m_PSOCaches;

    RefCntAutoPtr<ITextureView> m_pWhiteTexSRV;
    RefCntAutoPtr<ITextureView> m_pBlackTexSRV;
    RefCntAutoPtr<ITextureView> m_pDefaultNormalMapSRV;
    RefCntAutoPtr<ITextureView> m_pDefaultPhysDescSRV;

    static constexpr TEXTURE_FORMAT IrradianceCubeFmt = TEX_FORMAT_RGBA32_FLOAT;
    static constexpr TEXTURE_FORMAT PrefilteredEnvMapFmt = TEX_FORMAT_RGBA16_FLOAT;
    static constexpr Uint32 IrradianceCubeDim = 64;
    static constexpr Uint32 PrefilteredEnvMapDim = 256;

    RefCntAutoPtr<ITextureView> m_pIrradianceCubeSRV;
    RefCntAutoPtr<ITextureView> m_pPrefilteredEnvMapSRV;
    RefCntAutoPtr<IPipelineState> m_pPrecomputeIrradianceCubePSO;
    RefCntAutoPtr<IPipelineState> m_pPrefilteredEnvMapPSO;
    RefCntAutoPtr<IShaderResourceBinding> m_pPrescomputeIrradianceCubeSRB;
    RefCntAutoPtr<IShaderResourceBinding> m_pPrefilteredEnvMapSRB;

    RenderInfo m_RenderParams;

    RefCntAutoPtr<IBuffer> m_TransformCB;
    RefCntAutoPtr<IBuffer> m_GTLFAttribCB;
    RefCntAutoPtr<IBuffer> m_PrecomputeEnvMapAttribsCB;
    RefCntAutoPtr<IBuffer> m_JointsBuffer;
};

DEFINE_FLAG_ENUM_OPERATORS(QxGLTF_PBR_Render::RenderInfo::ALPHA_MODE_FLAGS)

}