struct ShaderIncInfo
{
    const char* const FileName;
    const char* const Source;
};

static const ShaderIncInfo g_Shaders[] =
{
    {
        "FullScreenTriangleVS.fx",
        #include "FullScreenTriangleVS.fx.h"
    },
    {
        "FullScreenTriangleVSOutput.fxh",
        #include "FullScreenTriangleVSOutput.fxh.h"
    },
    {
        "QxFullScreenTriangleVS.hlsl",
        #include "QxFullScreenTriangleVS.hlsl.h"
    },
    {
        "QxFullScreenTriangleVSOutput.hlsl",
        #include "QxFullScreenTriangleVSOutput.hlsl.h"
    },
    {
        "BasicStructures.fxh",
        #include "BasicStructures.fxh.h"
    },
    {
        "PBR_Common.fxh",
        #include "PBR_Common.fxh.h"
    },
    {
        "QxBasicStructures.hlsl",
        #include "QxBasicStructures.hlsl.h"
    },
    {
        "QxPBR_Common.hlsl",
        #include "QxPBR_Common.hlsl.h"
    },
    {
        "QxShaderUtilities.hlsl",
        #include "QxShaderUtilities.hlsl.h"
    },
    {
        "QxShadows.hlsl",
        #include "QxShadows.hlsl.h"
    },
    {
        "ShaderUtilities.fxh",
        #include "ShaderUtilities.fxh.h"
    },
    {
        "Shadows.fxh",
        #include "Shadows.fxh.h"
    },
    {
        "ComputeIrradianceMap.psh",
        #include "ComputeIrradianceMap.psh.h"
    },
    {
        "CubemapFace.vsh",
        #include "CubemapFace.vsh.h"
    },
    {
        "GLTF_PBR_PrecomputeCommon.fxh",
        #include "GLTF_PBR_PrecomputeCommon.fxh.h"
    },
    {
        "PrecomputeGLTF_BRDF.psh",
        #include "PrecomputeGLTF_BRDF.psh.h"
    },
    {
        "PrefilterEnvMap.psh",
        #include "PrefilterEnvMap.psh.h"
    },
    {
        "QxComputeIrradianceMapPS.hlsl",
        #include "QxComputeIrradianceMapPS.hlsl.h"
    },
    {
        "QxCubemapFaceVS.hlsl",
        #include "QxCubemapFaceVS.hlsl.h"
    },
    {
        "QxGLTF_PBR_PrecomputeCommon.hlsl",
        #include "QxGLTF_PBR_PrecomputeCommon.hlsl.h"
    },
    {
        "QxPrecomputeGLTF_BRDF_PS.hlsl",
        #include "QxPrecomputeGLTF_BRDF_PS.hlsl.h"
    },
    {
        "QxPrefilterEnvMapPS.hlsl",
        #include "QxPrefilterEnvMapPS.hlsl.h"
    },
    {
        "QxRenderGLTF_PBR_PS.hlsl",
        #include "QxRenderGLTF_PBR_PS.hlsl.h"
    },
    {
        "QxRenderGLTF_PBR_VS.hlsl",
        #include "QxRenderGLTF_PBR_VS.hlsl.h"
    },
    {
        "RenderGLTF_PBR.psh",
        #include "RenderGLTF_PBR.psh.h"
    },
    {
        "RenderGLTF_PBR.vsh",
        #include "RenderGLTF_PBR.vsh.h"
    },
    {
        "GLTF_PBR_Shading.fxh",
        #include "GLTF_PBR_Shading.fxh.h"
    },
    {
        "GLTF_PBR_Structures.fxh",
        #include "GLTF_PBR_Structures.fxh.h"
    },
    {
        "GLTF_PBR_VertexProcessing.fxh",
        #include "GLTF_PBR_VertexProcessing.fxh.h"
    },
    {
        "QxGLTF_PBR_Shading.hlsl",
        #include "QxGLTF_PBR_Shading.hlsl.h"
    },
    {
        "QxGLTF_PBR_Structures.hlsl",
        #include "QxGLTF_PBR_Structures.hlsl.h"
    },
    {
        "QxGLTF_PBR_VertexProcessing.hlsl",
        #include "QxGLTF_PBR_VertexProcessing.hlsl.h"
    },
    {
        "AtmosphereShadersCommon.fxh",
        #include "AtmosphereShadersCommon.fxh.h"
    },
    {
        "CoarseInsctr.fx",
        #include "CoarseInsctr.fx.h"
    },
    {
        "ComputeMinMaxShadowMapLevel.fx",
        #include "ComputeMinMaxShadowMapLevel.fx.h"
    },
    {
        "Extinction.fxh",
        #include "Extinction.fxh.h"
    },
    {
        "InitializeMinMaxShadowMap.fx",
        #include "InitializeMinMaxShadowMap.fx.h"
    },
    {
        "InterpolateIrradiance.fx",
        #include "InterpolateIrradiance.fx.h"
    },
    {
        "LookUpTables.fxh",
        #include "LookUpTables.fxh.h"
    },
    {
        "MarkRayMarchingSamples.fx",
        #include "MarkRayMarchingSamples.fx.h"
    },
    {
        "RayMarch.fx",
        #include "RayMarch.fx.h"
    },
    {
        "ReconstructCameraSpaceZ.fx",
        #include "ReconstructCameraSpaceZ.fx.h"
    },
    {
        "RefineSampleLocations.fx",
        #include "RefineSampleLocations.fx.h"
    },
    {
        "RenderCoordinateTexture.fx",
        #include "RenderCoordinateTexture.fx.h"
    },
    {
        "RenderSampling.fx",
        #include "RenderSampling.fx.h"
    },
    {
        "RenderSliceEndPoints.fx",
        #include "RenderSliceEndPoints.fx.h"
    },
    {
        "ScatteringIntegrals.fxh",
        #include "ScatteringIntegrals.fxh.h"
    },
    {
        "SliceUVDirection.fx",
        #include "SliceUVDirection.fx.h"
    },
    {
        "Sun.fx",
        #include "Sun.fx.h"
    },
    {
        "UnshadowedScattering.fxh",
        #include "UnshadowedScattering.fxh.h"
    },
    {
        "UnwarpEpipolarScattering.fx",
        #include "UnwarpEpipolarScattering.fx.h"
    },
    {
        "UpdateAverageLuminance.fx",
        #include "UpdateAverageLuminance.fx.h"
    },
    {
        "CombineScatteringOrders.fx",
        #include "CombineScatteringOrders.fx.h"
    },
    {
        "ComputeScatteringOrder.fx",
        #include "ComputeScatteringOrder.fx.h"
    },
    {
        "ComputeSctrRadiance.fx",
        #include "ComputeSctrRadiance.fx.h"
    },
    {
        "InitHighOrderScattering.fx",
        #include "InitHighOrderScattering.fx.h"
    },
    {
        "PrecomputeAmbientSkyLight.fx",
        #include "PrecomputeAmbientSkyLight.fx.h"
    },
    {
        "PrecomputeCommon.fxh",
        #include "PrecomputeCommon.fxh.h"
    },
    {
        "PrecomputeNetDensityToAtmTop.fx",
        #include "PrecomputeNetDensityToAtmTop.fx.h"
    },
    {
        "PrecomputeSingleScattering.fx",
        #include "PrecomputeSingleScattering.fx.h"
    },
    {
        "UpdateHighOrderScattering.fx",
        #include "UpdateHighOrderScattering.fx.h"
    },
    {
        "EpipolarLightScatteringFunctions.fxh",
        #include "EpipolarLightScatteringFunctions.fxh.h"
    },
    {
        "EpipolarLightScatteringStructures.fxh",
        #include "EpipolarLightScatteringStructures.fxh.h"
    },
    {
        "QxToneMapping.hlsl",
        #include "QxToneMapping.hlsl.h"
    },
    {
        "QxToneMappingStructures.hlsl",
        #include "QxToneMappingStructures.hlsl.h"
    },
    {
        "ToneMapping.fxh",
        #include "ToneMapping.fxh.h"
    },
    {
        "ToneMappingStructures.fxh",
        #include "ToneMappingStructures.fxh.h"
    },
    {
        "ShadowConversions.fx",
        #include "ShadowConversions.fx.h"
    },
};
