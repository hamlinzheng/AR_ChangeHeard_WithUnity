using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System;
// using System;
using System.Text;
// using System.Net.Sockets;
// using System.Net;



public class SocketServer : MonoBehaviour
{
    public static string data;

    System.Threading.Thread SocketThread;
    volatile bool keepReading = false;
    public int wide, height;
    private byte[] pbytes;

    // Use this for initialization
    void Start()
    {
        Application.runInBackground = true;  
        pbytes = ReadImageFile("2.png");
    }

    public static void Execute(string command)// Execute cmd command
    {
        var processInfo = new System.Diagnostics.ProcessStartInfo("cmd.exe", "/S /C " + command)
        {
            CreateNoWindow = true,
            UseShellExecute = true,
            WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden
        };

        System.Diagnostics.Process.Start(processInfo);
    }

    public void StartServer()
    {
        SocketThread = new System.Threading.Thread(NetworkCode);
        SocketThread.IsBackground = true;
        SocketThread.Start();
        
        // Execute("cd PythonScript & python visual_measurement.py");// Execute python client script
    }



    private string GetIPAddress()
    {
        string localIP = "127.0.0.1";
        return localIP;
    }

    private static byte[] ReadImageFile(String img)
    {
      FileInfo fileinfo = new FileInfo (img);
      byte[] buf = new byte[fileinfo.Length];
      FileStream fs = new FileStream (img, FileMode.Open, FileAccess.Read);
      fs.Read (buf, 0, buf.Length);
      fs.Close ();
      //fileInfo.Delete ();
      GC.ReRegisterForFinalize (fileinfo);
      GC.ReRegisterForFinalize (fs);
      return buf;
    }
    Socket listener;
    Socket handler;

    void NetworkCode()
    {
        // Data buffer for incoming data.
        byte[] bytes = new Byte[1024];

        // Host running the application.
        Debug.Log("Ip " + GetIPAddress().ToString());
        IPAddress[] ipArray = Dns.GetHostAddresses(GetIPAddress());
        IPEndPoint localEndPoint = new IPEndPoint(ipArray[0], 1755);//端口为1755

        // Create a TCP/IP socket.
        listener = new Socket(ipArray[0].AddressFamily,
            SocketType.Stream, ProtocolType.Tcp);

        // Bind the socket to the local endpoint and 
        // listen for incoming connections.

        try
        {
            listener.Bind(localEndPoint);
            listener.Listen(10);

            // Start listening for connections.
            while (true)
            {
                keepReading = true;

                // Program is suspended while waiting for an incoming connection.
                Debug.Log("Waiting for Connection");
                
                
                handler = listener.Accept();
                Debug.Log("Client Connected");
                data = null;
                
                // An incoming connection needs to be processed.
                while (keepReading)
                {
                    bytes = new byte[1024];
                    int bytesRec = handler.Receive(bytes);
                    Debug.Log("Received from Server");

                    // Debug.Log("W second   " + System.DateTime.Now.Second); // 当前时间(秒)
                    if (bytesRec <= 0)
                    {
                        keepReading = false;
                        handler.Disconnect(true);
                        break;
                    }

                    data = Encoding.ASCII.GetString(bytes, 0, bytesRec);
                    
                    string[] tempdata = data.Split(':');

                    if (data.IndexOf("<EOF>") > -1)
                    {
                        break;
                    }

                    System.Threading.Thread.Sleep(1);
                    // byte[] pData = System.Text.Encoding.Default.GetBytes("server : test");
                    // Debug.Log("Send"+CameraCapture.bytes);
                    handler.Send(CropCameraImg.bytes);//给客户端发消息
                    // Debug.Log("Data Send succuss");
//                     byte[] buffer = ReadImageFile ("1.jpg");
//                     handler.Send (buffer, buffer.Length, SocketFlags.None);
                    
                }
                
                System.Threading.Thread.Sleep(20);
            }
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }

    void StopServer()
    {
        keepReading = false;

        //stop thread
        if (SocketThread != null)
        {
            //listener.Shutdown(SocketShutdown.Both);
            //listener.Close();
            SocketThread.Abort();
        }

        if (handler != null && handler.Connected)
        {
            handler.Disconnect(false);
            Debug.Log("Disconnected!");
        }
    }

    public void OnDisable()
    {
        StopServer();
    }

    private void Update() {
        wide = Screen.width;
        height = Screen.height;
    }
    // private void OnPostRender() {
    //     Debug.Log("Test !");
    //     // CaptureScreenshot2( new Rect( Screen.width*0f, Screen.height*0f, Screen.width*1f, Screen.height*1f));
    // }
    

}

