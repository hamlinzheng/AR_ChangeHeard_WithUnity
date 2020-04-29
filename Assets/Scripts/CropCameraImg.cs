using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
// using UnityEngine;
using System.IO;

public class CropCameraImg : MonoBehaviour
{
    public Camera cropCamera; //待截图的目标摄像机
    public static Texture2D img;
    public static byte[] bytes;
    RenderTexture renderTexture;
    Texture2D texture2D;
    // Start is called before the first frame update
    void Start()
    {

        // renderTexture = new RenderTexture(Screen.width, Screen.height, 32);
        texture2D = new Texture2D(Screen.width, Screen.height, TextureFormat.ARGB32, false);
        // cropCamera.targetTexture = renderTexture; 
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    private void OnPostRender() {
        // RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(Screen.width*0f, Screen.height*0f, Screen.width*1f, Screen.height*1f), 0, 0);
        texture2D.Apply();
        // RenderTexture.active = null;
        
        // byte[] pbytes = texture2D.EncodeToPNG();
        // File.WriteAllBytes(Application.dataPath + "//pic//" + (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, 0)).TotalMilliseconds + ".png", pbytes);
        bytes = texture2D.EncodeToPNG();
    }
}
