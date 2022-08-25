#include <cstring>
#include <array>

#include "QxGLTF_PBR_Render.hpp"
#include "../../../Utilities/include/DiligentFXShaderSourceStreamFactory.hpp"
#include "CommonlyUsedStates.h"
#include "HashUtils.hpp"
#include "ShaderMacroHelper.hpp"
#include "BasicMath.hpp"
#include "GraphicsUtilities.h"
#include "MapHelper.hpp"
#include "GraphicsAccessories.hpp"

namespace Diligent
{
QxGLTF_PBR_Render::QxGLTF_PBR_Render(IRenderDevice* pDevice, IDeviceContext* pCtx, const CreateInfo& CI)
    : m_Settings(CI)
{
    // 准备IBL用的纹理的响应的srv
    if (m_Settings.UseIBL)
    {
        PrecomputeBRDF(pDevice, pCtx);

        TextureDesc TexDesc;
        TexDesc.Name = "Irradiance cube map for GLTF render";
        TexDesc.Type = RESOURCE_DIM_TEX_CUBE;
        TexDesc.Usage = USAGE_DEFAULT;
        TexDesc.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET;
        TexDesc.Width = IrradianceCubeDim;
        TexDesc.Height = IrradianceCubeDim;
        TexDesc.Format = IrradianceCubeFmt;
        TexDesc.ArraySize = 6;
        TexDesc.MipLevels = 0;

        RefCntAutoPtr<ITexture> IrradianceCubeTex;
        pDevice->CreateTexture(TexDesc, nullptr,
            &IrradianceCubeTex);
        m_pIrradianceCubeSRV =
            IrradianceCubeTex->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Prefiltered environment map for GLTF renderer";
        TexDesc.Width = PrefilteredEnvMapDim;
        TexDesc.Height = PrefilteredEnvMapDim;
        TexDesc.Format = PrefilteredEnvMapFmt;
        RefCntAutoPtr<ITexture> PrefilteredEnvMapTex;
        pDevice->CreateTexture(TexDesc, nullptr, &PrefilteredEnvMapTex);
        m_pPrefilteredEnvMapSRV =
            PrefilteredEnvMapTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    // 创建和初始化black/white/defaultNromalmap/default physic 这几个纹理和响应的SRV/Sampler
    {
        static constexpr Uint32 TexDim = 8;
        TextureDesc TexDesc;
        TexDesc.Name = "White texture for GLTF renderer";
        TexDesc.Type = RESOURCE_DIM_TEX_2D_ARRAY;
        TexDesc.Usage = USAGE_IMMUTABLE;
        TexDesc.BindFlags = BIND_SHADER_RESOURCE;
        TexDesc.Width = TexDim;
        TexDesc.Height = TexDim;
        TexDesc.Format = TEX_FORMAT_RGBA8_UNORM;
        TexDesc.MipLevels = 1;
        std::vector<Uint32> Data(TexDim * TexDim, 0xFFFFFFFF);
        TextureSubResData Level0Data{Data.data(), TexDim * 4};
        TextureData InitData{&Level0Data, 1};
        RefCntAutoPtr<ITexture> pWhiteTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pWhiteTex);
        m_pWhiteTexSRV = pWhiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Black texture for GLTF renderer";
        for (Uint32& c : Data)
        {
            c = 0;
        }
        RefCntAutoPtr<ITexture> pBlackTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pBlackTex);
        m_pBlackTexSRV = pBlackTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Default normal map for GLTF renderer";
        for (Uint32& color : Data)
        {
            color = 0x00FF7F7F;
        }
        RefCntAutoPtr<ITexture> pDefaultNormalMap;
        pDevice->CreateTexture(TexDesc, &InitData, &pDefaultNormalMap);
        m_pDefaultNormalMapSRV =
            pDefaultNormalMap->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Default physical descriptioin map for GLTF renderer";
        for (Uint32& color : Data)
        {
            color = 0x0000FF00;
        }
        RefCntAutoPtr<ITexture> pDefaultPhycDescTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pDefaultPhycDescTex);
        m_pDefaultPhysDescSRV =
            pDefaultPhycDescTex->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        StateTransitionDesc Barriers[] =
        {
            {pWhiteTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pBlackTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pDefaultNormalMap, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pDefaultPhycDescTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
        };
        pCtx->TransitionResourceStates(_countof(Barriers), Barriers);

        RefCntAutoPtr<ISampler> pDefaultSampler;
        pDevice->CreateSampler(Sam_LinearClamp, &pDefaultSampler);
        m_pWhiteTexSRV->SetSampler(pDefaultSampler);
        m_pBlackTexSRV->SetSampler(pDefaultSampler);
        m_pDefaultNormalMapSRV->SetSampler(pDefaultSampler);
    }

    if (CI.RTVFmt != TEX_FORMAT_UNKNOWN || CI.DSVFmt != TEX_FORMAT_UNKNOWN)
    {
        CreateUniformBuffer(pDevice,
            sizeof(GLTFNodeShaderTransforms),
            "GLTF node transform CB",
            &m_TransformCB);
        CreateUniformBuffer(pDevice,
            sizeof(GLTFMaterialShaderInfo) + sizeof(GLTFRendererShaderParameters),
            "GLTF attribs CB",
            &m_GTLFAttribCB);
    }
}

void QxGLTF_PBR_Render::Render(IDeviceContext* pCtx,
                               GLTF::Model& GLTFModel,
                               const RenderInfo& RenderParams,
                               ModelResourceBindings* pModelBindings,
                               ResourceCacheBindings* pCacheBindings)
{
    DEV_CHECK_ERR((pModelBindings != nullptr) && (pCacheBindings != nullptr),
    "Either model bindings or cache bindings must not be null");
    DEV_CHECK_ERR(pModelBindings == nullptr ||
        pModelBindings->MaterialSRB.size() == GLTFModel.Materials.size(),
        "The number of material shader resource bindings is not consistent with the number of materials");

    // 设置vetex buffer和index buffer
    if (pModelBindings != nullptr)
    {
        std::array<IBuffer*, 2> pVBs =
            {
                GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_VERTEX_BASIC_ATTRIBS),
                GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_VERTEX_SKIN_ATTRIBS)
            };
        pCtx->SetVertexBuffers(0,
            static_cast<Uint32>(pVBs.size()),
                pVBs.data(),
                nullptr,
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                SET_VERTEX_BUFFERS_FLAG_RESET);

        if (auto* pIndexBuffer = GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_INDEX))
        {
            pCtx->SetIndexBuffer(
                pIndexBuffer,
                0,
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
    }

    const auto ModelFirstIndexLocation = GLTFModel.GetFirstIndexLocation();
    const auto ModelBaseVertex = GLTFModel.GetBaseVertex();

    const std::array<GLTF::Material::ALPHA_MODE, 3> AlphaModes
    {
        GLTF::Material::ALPHA_MODE_OPAQUE,// Opaque primitives - first
        GLTF::Material::ALPHA_MODE_MASK,// Alpha-masked primitives - second
        GLTF::Material::ALPHA_MODE_BLEND// Transparent primitives - last (TODO: depth sorting)
    };

    const GLTF::Mesh* pLastAnimationMesh = nullptr;
    IPipelineState* pCurPSO = nullptr;
    IShaderResourceBinding* pCurSRB = nullptr;
    PSOKey CurrPSOKey;

    for (GLTF::Material::ALPHA_MODE AlphaMode : AlphaModes)
    {
        for (const GLTF::Node* pNode : GLTFModel.LinearNodes)
        {
            if (!pNode->pMesh)
            {
                continue;
            }

            const GLTF::Mesh& Mesh = *pNode->pMesh;

            // Render mesh primitives
            for (const GLTF::Primitive& primitive : Mesh.Primitives)
            {
                const GLTF::Material& material =
                    GLTFModel.Materials[primitive.MaterialId];
                if (material.Attribs.AlphaMode != AlphaMode)
                {
                    continue;
                }

                // 根据需要更新和设置当前pso,以及清空SRB
                {
                const PSOKey Key{AlphaMode, material.DoubleSided};
                if (Key != CurrPSOKey)
                {
                    CurrPSOKey = Key;
                    pCurPSO = nullptr;
                }
                if (pCurPSO == nullptr)
                {
                    pCurPSO = GetPSO(CurrPSOKey);
                    VERIFY_EXPR(pCurPSO != nullptr);
                    pCtx->SetPipelineState(pCurPSO);
                    pCurSRB = nullptr;
                }
                else
                {
                    VERIFY_EXPR(pCurPSO ==
                        GetPSO(PSOKey{AlphaMode, material.DoubleSided}));
                }
                    
                }


                // 获得draw call要用的SRB并提交
                if (pModelBindings != nullptr)
                {
                    VERIFY(primitive.MaterialId <
                        pModelBindings->MaterialSRB.size(),
                           "Material index is out of bounds. This mostl likely indicates that shader resources were initialized for a different model.");

                    IShaderResourceBinding* const pSRB =
                        pModelBindings->MaterialSRB[primitive.MaterialId]
                            .RawPtr<IShaderResourceBinding>();
                    DEV_CHECK_ERR(pSRB != nullptr, "Unable to find SRB GLTF material");
                    if (pCurSRB != pSRB)
                    {
                        pCurSRB = pSRB;
                        pCtx->CommitShaderResources(
                            pSRB,
                            RESOURCE_STATE_TRANSITION_MODE_VERIFY);
                    }
                }
                else
                {
                    VERIFY_EXPR(pCacheBindings != nullptr);
                    if (pCurSRB != pCacheBindings->pSRB)
                    {
                        pCurSRB = pCacheBindings->pSRB;
                        pCtx->CommitShaderResources(
                            pCurSRB,
                            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                    }
                }

#pragma region UpdateShaderParameters
                size_t JointCount = Mesh.Transforms.jointMatrices.size();
                if (JointCount > m_Settings.MaxJointCount)
                {
                    LOG_WARNING_MESSAGE("The number of joints in the mesh (", JointCount, ") exceeds the maximum number (", m_Settings.MaxJointCount,
                                    ") reserved in the buffer. Increase MaxJointCount when initializing the renderer.");
                    JointCount = m_Settings.MaxJointCount;
                }

                // 上传node matrix /joint count参数
                {
                    MapHelper<GLTFNodeShaderTransforms> pTransforms{
                        pCtx, m_TransformCB,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };
                    pTransforms->NodeMatrix =
                        Mesh.Transforms.matrix * RenderParams.ModelTransform;
                    pTransforms->JointCount =
                        static_cast<int>(JointCount);
                }

                // 上传joints matrixes 到gpu
                if (JointCount != 0 && pLastAnimationMesh != &Mesh)
                {
                    MapHelper<float4x4> pJoints{
                        pCtx, m_JointsBuffer,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };
                    memcpy(pJoints, Mesh.Transforms.jointMatrices.data(),
                        JointCount * sizeof(float4x4));
                    pLastAnimationMesh = &Mesh;
                }

                // 上传材质属性参数和renderer的参数到gpu
                {
                    // 这个用来辅助上传数据到gpu
                    struct GLTFAttribs
                    {
                        GLTFRendererShaderParameters RenderParameters;
                        GLTF::Material::ShaderAttribs MaterialInfo;
                        static_assert(
                            sizeof(GLTFMaterialShaderInfo) == sizeof(GLTF::Material::ShaderAttribs),
                            "The sizeof(GLTFMaterialShaderInfo) is inconsistent with sizeof(GLTF::Material::ShaderAttribs)");
                    };
                    static_assert(
                        sizeof(GLTFAttribs) <= 256,
                        "Size of dynamic GLTFAttribs buffer exceeds 256 bytes. "
                                                              "It may be worth trying to reduce the size or just live with it.");
                    MapHelper<GLTFAttribs> pGLTFAttribs{
                        pCtx, m_GTLFAttribCB,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };

                    pGLTFAttribs->MaterialInfo = material.Attribs;

                    auto& ShaderParams = pGLTFAttribs->RenderParameters;

                    ShaderParams.DebugViewType =
                        static_cast<int>(m_RenderParams.DebugView);
                    ShaderParams.OcclusionStrength =
                        m_RenderParams.OcclusionStrength;
                    ShaderParams.EmissionScale =
                        m_RenderParams.EmissionScale;
                    ShaderParams.AverageLogLum = m_RenderParams.AverageLogLum;
                    ShaderParams.MiddleGray =
                        m_RenderParams.MiddleGray;
                    ShaderParams.WhitePoint = m_RenderParams.WhitePoint;
                    ShaderParams.IBLScale = m_RenderParams.IBLScale;
                    
                }

#pragma endregion

                // 进行draw call
                if (primitive.HasIndices())
                {
                    DrawIndexedAttribs drawAttrs{
                        primitive.IndexCount,
                        VT_UINT32,
                        DRAW_FLAG_VERIFY_ALL
                        };
                    drawAttrs.FirstIndexLocation =
                        ModelFirstIndexLocation + primitive.FirstIndex;
                    drawAttrs.BaseVertex = ModelBaseVertex;
                    pCtx->DrawIndexed(drawAttrs);
                }
                else
                {
                    DrawAttribs drawAttrs{
                        primitive.VertexCount,
                        DRAW_FLAG_VERIFY_ALL
                        };
                    drawAttrs.StartVertexLocation = ModelBaseVertex;
                    pCtx->Draw(drawAttrs);
                }
            }
        }
    }
}
}
