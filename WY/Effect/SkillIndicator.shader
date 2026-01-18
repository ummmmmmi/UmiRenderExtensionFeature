Shader "WangYue/VFX/SkillIndicator"
{
    Properties
    {
        //基础设置
        [Enum(UnityEngine.Rendering.CullMode)]_CullMode("剔除模式",int) = 2
        [Toggle(_ENABLE_FOG_TRANS)] _enableFog("启用场景雾", int) = 0

        [Space(20)]
        //主纹理
        [Toggle(_TWOFACECOLOR_ON)] _EnableTwoSidedCol("是否启用双面颜色", float) = 0
        _MainTex ("主纹理贴图", 2D) = "white" {}
        [HDR]_MainColor ("主纹理颜色", Color) = (1,1,1,1)
        [HDR]_BackColor ("主纹理背面颜色", Color) = (1,1,1,1)
        [Enum(RGB,0,R,1,G,2,B,3,A,4)]_MainChannel("主纹理颜色通道选择", float) = 0
        [Enum(A,0,R,1,G,2,B,3,None,4)]_MainAlphaChannel("主纹理透明通道选择",float) = 0
        
        [Space(20)]
        //形状选择
        [Enum(Rect, 0, Circle, 1, Sector, 2)]
        _ShapeType("指示器形状", int) = 2
        
        //通用
        [Toggle(_ROTATION_ON)] _EnableRotate("是否使用旋转",float) = 0
        _RotateAngle("指示器旋转", Range(0, 360)) = 0.0
        _Radius("圆形半径/扇形半径", Range(0, 1)) = 0.5
        [Enum(OneSide,0,OneSide_Inv,1,TwoSide,2,TwoSide_Inv,3,SDF,4)]_SweepFlowMode("矩形流动方向",float) = 3

        [Space(20)]
        //外边框
        [HDR]_IndicatorEdgeColor("指示器外边框颜色", Color) = (1,1,1,1)
        _IndicatorEdgeWidth("指示器外边框宽度", Range(0.001, 0.2)) = 0.01
        
        [HDR]_IndicatorEdgeFadeColor("指示器边缘渐变颜色", Color) = (1,1,1,1)
        _IndicatorEdgeFadeMin("指示器边缘渐变范围", Range(0.00, 1)) = 0.5
        _IndicatorEdgeFadeMax("指示器边缘渐变过渡", Range(0.00, 1)) = 0.5
        
        [Space(20)]
        //扇形
        _SectorAngle("扇形打开角度", Range(0, 360)) = 45
    }
    SubShader
    {
        Tags
        {
            "IgnoreProjector" = "True"
            "Queue" = "Transparent"
            "RenderType" = "Transparent"
        }

        Pass
        {
            Tags
            {
                "RenderPipeline" = "UniversalPipeline"
                "LightMode" = "CharacterTransparent"
            }
            
            Cull [_CullMode]
            Blend SrcAlpha OneMinusSrcAlpha
            Zwrite off

            HLSLPROGRAM
            #pragma target 4.5
            // #pragma multi_compile_fog //内置雾

            // #pragma shader_feature_local  _ _ENABLE_FOG_TRANS //雾效宏
            #pragma shader_feature_local_fragment  _ _TWOFACECOLOR_ON//双面颜色
            #pragma shader_feature_local_fragment  _ _ROTATION_ON   //旋转宏
            // #pragma shader_feature_local_fragment  _ _SHAPE_RECT _SHAPE_CIRCLE _SHAPE_SECTOR   //形状宏

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            CBUFFER_START(UnityPerMaterial)
            float4 _MainTex_ST;
            half4 _MainColor;
            half4 _BackColor;
            half _MainChannel;
            half _MainAlphaChannel;

            int _ShapeType;
            int _SweepFlowMode;

            half _RotateAngle;
            half _Radius;
            half _SectorAngle;

            half4 _IndicatorEdgeColor;
            half _IndicatorEdgeWidth;

            half4 _IndicatorEdgeFadeColor;
            half _IndicatorEdgeFadeMin, _IndicatorEdgeFadeMax;
            
            CBUFFER_END

            #define PI_DIV_180 0.017453292f     // pi / 180
            #define ROT_180_MAT float2x2(-1.0f, 0.0f, 0.0f, -1.0f)  //旋转180度

            inline half3 GetChannelXYZ(half4 mask1, int ChannelId)
            {
                const half3 CHANNEL[5] =
                {
                    {mask1.x, mask1.y, mask1.z},
                    {mask1.x, mask1.x, mask1.x},
                    {mask1.y, mask1.y, mask1.y},
                    {mask1.z, mask1.z, mask1.z},
                    {mask1.w, mask1.w, mask1.w},

                };

                return CHANNEL[ChannelId];
            }

            inline half GetChannelMask(half4 maskColor, int ChannelId)
            {
                const half CHANNEL[5] =
                {
                    {maskColor.a},
                    {maskColor.r},
                    {maskColor.g},
                    {maskColor.b},
                    {1.0f},

                };

                return CHANNEL[ChannelId];
            }

            inline float GetRectSweepFlow(half fade, float2 uv, int ChannelId)
            {
                const float CHANNEL[5] =
                {
                    {(uv.y - 0.5)},
                    {0.5 - uv.y},
                    {abs(uv.y - 0.5)},
                    {0.5 - abs(uv.y - 0.5)},
                    {fade},
                };
                return CHANNEL[ChannelId];
            }
            
            inline float Deg2Rad(float degrees)
            {
                return degrees * PI_DIV_180;
            }

            inline float getSDFEdge(float sdfRange)
            {
                float outer = (sdfRange + _IndicatorEdgeWidth) > 0.0;
                float inner = sdfRange > 0.0;
                return outer - inner;
            }
            
            inline float sdPie(float2 pos, float2 c, float radius, out float distanceToCenter)
            {
                pos.x = abs(pos.x);
                distanceToCenter = length(pos);
                float l = distanceToCenter - radius;
                float m = length(pos - c * clamp(dot(pos,c), 0.0, radius) );
                return max(l, m * sign(c.y * pos.x - c.x * pos.y));
            }

            // 修复了大于180度时，圆心边缘线呈圆角
            inline float sdPie_fixed(float2 pos, float angle, float radius, float edgeWidth,
                out float distanceToCenter, out float shapeMask, out float edgeMask)
            {
                angle = angle * 0.5;
                bool isOver90 = angle > 90;
                pos = isOver90 ? mul(pos, ROT_180_MAT) : pos;
                
                distanceToCenter = length(pos);
                half edgeRadius = radius;
                half maskRadius = radius;
                
                if (isOver90)
                {
                    angle = clamp(0.0, 360.0, 180 - angle);
                    edgeRadius -= edgeWidth;
                    radius += 0.5;
                }
                else
                {
                    angle = clamp(0.0, 360.0, angle);
                }
                
                pos.x = abs(pos.x);
                half l = distanceToCenter - radius;    //circle
                half rad = Deg2Rad(angle);
                half2 c = half2(sin(rad), cos(rad));
                float m = length(pos - c * clamp(dot(pos,c), 0.0, radius) );
                float signRange = sign(c.y * pos.x - c.x * pos.y);
                float sector = max(l, m * signRange);

                //edgeShape
                float edgeCircle = distanceToCenter - edgeRadius;
                float shape = isOver90 ? min(-edgeCircle, sector) : sector;
                edgeMask = getSDFEdge(shape) * _IndicatorEdgeColor.a;

                //maskShape
                float maskCircle = distanceToCenter - maskRadius;
                float maskShape = isOver90 ? min(-maskCircle, sector) : sector;
                shapeMask = maskShape > 0 ? 1.0 : 0.0;
                shapeMask = isOver90 ? (1 - shapeMask) : shapeMask;
                shapeMask = max(shapeMask, edgeMask);

                return shape;
            }

            inline float sdBox(float2 p, float2 b)
            {
                float2 d = abs(p) - b;
                return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
            }

            // inline float2 RotationUV(float2 uv, float angle)
            // {
            //     float sinNum = sin(angle);
            //     float cosNum = cos(angle);
            //     float2 rotaUV_center = mul(uv - 0.5f, float2x2(cosNum, -sinNum, sinNum, cosNum));
            //     return rotaUV_center;
            // }

            // 顶点阶段预计算旋转（片元阶段复用）
            inline half2 RotationUV(half2 uv, half2 sinCos)
            {
                return mul(uv - 0.5, float2x2(sinCos.y, -sinCos.x, sinCos.x, sinCos.y));
            }

            struct Attributes
            {
                float3 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                // half4 vertexColor : COLOR;
                // float2 uv2: TEXCOORD3;
            };

            struct Varyings
            {
                float4 pos : SV_POSITION;
                float4 uv : TEXCOORD0;
                half4 vertexColor : COLOR;
                float2 rotatedUV : TEXCOORD2;

                #if _Fresnel_ON || _CubeMap_ON || _ENABLE_FOG_TRANS
                float3 worldPos : TEXCOORD7;
                float3 worldNormal : TEXCOORD8;
                #endif

                float4 screenPos : TEXCOORD9;
                float3 posOS : TEXCOORD10;
            };

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                output.pos = TransformObjectToHClip(input.positionOS);
                output.uv.xy = input.uv;
                // output.uv.zw = input.uv2;
                
                #ifdef _ROTATION_ON
                half rad = Deg2Rad(_RotateAngle);
                half2 sinCos = half2(sin(rad), cos(rad));
                output.rotatedUV = sinCos;
                #else
                output.rotatedUV = 0;
                #endif
                
                return output;
            }

            half4 frag(Varyings input, bool vface : SV_IsFrontFace) : SV_TARGET
            {
                half4 finalColor;

                // 通用数据
                float2 mainuv = input.uv.xy;
                float2 shapeCenterPos = mainuv - 0.5;
                #ifdef _ROTATION_ON
                shapeCenterPos = RotationUV(mainuv, input.rotatedUV);
                #endif
                float sweepFade = 0;
                
                //形状计算
                float indicatorShape = 0;
                half indicatorMask = 0;
                half edge = 0;
                if (_ShapeType == 2)
                {
                    indicatorShape = sdPie_fixed(shapeCenterPos, _SectorAngle, _Radius, _IndicatorEdgeWidth,
                        sweepFade, indicatorMask, edge);
                }
                else if (_ShapeType == 1)
                {
                    sweepFade = length(shapeCenterPos);
                    indicatorShape = sweepFade - _Radius;
                    indicatorMask = indicatorShape < 0 ? 1.0 : 0.0;
                    
                    edge = getSDFEdge(indicatorShape) * _IndicatorEdgeColor.a;
                }
                else if (_ShapeType == 0)
                {
                    indicatorShape = sdBox(shapeCenterPos, half2(_Radius, _Radius));
                    sweepFade = GetRectSweepFlow(indicatorShape + 0.5, shapeCenterPos + 0.5, _SweepFlowMode);
                    indicatorMask = indicatorShape < 0 ? 1.0 : 0.0;
                    
                    edge = getSDFEdge(indicatorShape) * _IndicatorEdgeColor.a;
                }

                //扫光 主纹理
                float2 colMainTex_UV = float2(0.5, saturate(sweepFade - _MainTex_ST.w));
                half4 colMainTex = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, colMainTex_UV);
                half3 colMain = GetChannelXYZ(colMainTex, _MainChannel);
                half colMainA = GetChannelMask(colMainTex, _MainAlphaChannel);

                #ifdef _TWOFACECOLOR_ON
                float4 facecolor = (vface) ? (_MainColor) : (_BackColor);
                half3 col = colMain * facecolor.rgb;
                half alpha = saturate(colMainA * facecolor.a);
                #else
                half3 col = colMain * _MainColor.rgb;
                half alpha = saturate(colMainA * _MainColor.a);
                #endif

                //边缘过渡
                half edgeFade = smoothstep(_IndicatorEdgeFadeMin, _IndicatorEdgeFadeMin + _IndicatorEdgeFadeMax, sweepFade) * _IndicatorEdgeFadeColor.a;
                edgeFade *= (sweepFade < 0.5) ? 1.0 : 0.0;
                
                //混合颜色
                half3 colEdge = lerp(_IndicatorEdgeFadeColor.rgb * edgeFade, _IndicatorEdgeColor.rgb, edge);
                
                finalColor.rgb = max(colEdge, col * _MainColor.a);
                finalColor.a = saturate((_ShapeType == 2) ?
                    lerp(alpha + edgeFade, edge, indicatorMask) :
                    ((alpha + edgeFade + edge) * indicatorMask));
                
                return finalColor;
            }
            ENDHLSL
        }
    }
}