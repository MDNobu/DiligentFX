#ifndef __QxGLTF_PBR_Common
#define __QxGLTF_PBR_Common

#ifndef PI
#define PI 3.141592653589793
#endif

// Lambertian diffuse
float3 LambertianDiffuse(float3 DiffuseColor)
{
    return DiffuseColor / PI;
}

float3 SchlickFresnel(float VoH, float3 Reflectance0, float3 Reflecttance90)
{
    float res = Reflectance0 + (Reflecttance90 - Reflectance0) *
        pow(clamp(1 - VoH, 0.f, 1.f), 5.f);
    return res;
}

struct AngularInfo
{
    float NoL;
    float NoV;
    float NoH;
    float LoH;
    float VoH;
};

AngularInfo GetAngularInfo(float3 PointToLight,
    float3 Normal,
    float3 View)
{
    float3 n = normalize(Normal);
    float3 v = normalize(View);
    float3 l = normalize(PointToLight);
    float3 h = normalize(l + v);

    AngularInfo info;
    info.NoL = saturate(dot(n, l));
    info.NoV = saturate(dot(n, v));
    info.NoH = saturate(dot(n, h));
    info.LoH = saturate(dot(l, h));
    info.VoH = saturate(dot(v, h));
    return info;
}

struct QxSurfaceReflectanceInfo
{
    float PerceptualRoughness;
    float3 Reflectance0;
    float3 Reflectance90;
    float3 DiffuseColor;
};

//下面的公式参照UE4 的pbr文档
float NormalDistribution_GGX(float NoH, float AlphaRoughness)
{
    float a2 = AlphaRoughness * AlphaRoughness;
    float f = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * f * f);
}

// Visibility = G(v,l,a) / (4 * (n,v) * (n,l))
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float SmithGGXVisibilityCorrelated(float NoL, float NoV, float AlphaRoughness)
{
    float a2 = AlphaRoughness * AlphaRoughness;

    float GGXV = NoL * sqrt(max(NoV * NoV * (1 - a2) + a2, 1e-7));
    float GGXL = NoV * sqrt(max(NoL * NoL * (1 - a2) + a2, 1e-7));
    return 0.5 / (GGXL + GGXV);
}

void BRDF(
    in float3 PointToLight,
    in float3 Normal,
    in float3  View,
    in QxSurfaceReflectanceInfo SrfInfo,
    out float3 DiffuseContrib,
    out float3 SpecContrib,
    out float NoL)
{
    AngularInfo angularInfo = GetAngularInfo(PointToLight, Normal, View);

    DiffuseContrib = float3(0, 0, 0);
    SpecContrib = float3(0, 0, 0);
    NoL = angularInfo.NoL;

    if (angularInfo.NoL > 0 || angularInfo.NoV > 0)
    {
        float AlphaRoughness =
            SrfInfo.PerceptualRoughness * SrfInfo.PerceptualRoughness;
        float D = NormalDistribution_GGX(angularInfo.NoH, AlphaRoughness);
        float3 F = SchlickFresnel(angularInfo.VoH,
            SrfInfo.Reflectance0,
            SrfInfo.Reflectance90);
        float Vis = SmithGGXVisibilityCorrelated(
            angularInfo.NoL,
            angularInfo.NoV,
            AlphaRoughness
            );

        DiffuseContrib = (1.0 - F) * LambertianDiffuse(SrfInfo.DiffuseColor);
        SpecContrib = F * Vis * D;
    }
}

#endif
