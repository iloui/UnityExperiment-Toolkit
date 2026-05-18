using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.Networking;
using UnityTools.Components;
using System.Text.RegularExpressions;

namespace Assets.Scripts.DataRecording
{
    public class DataUploadHandler
    {
        private const int TargetDataPackageChars = 4 * 1024;

        /// <summary>
        /// Fixed format to prevent current culture to mess with the date time format (day/month/year hour:minute:second)
        /// </summary>
        private const string DateTimeFormat = "dd/MM/yyyy hh:mm:ss";

        [NotNull]
        private readonly string url;

        [NotNull]
        private readonly string floatPrecision;

        private string participantId;

        private string sessionId;
        private string expID;

        private int trialIndex;

        private float trialStartTime;

        private int trialPackageIndex;

        private Dictionary<string, int> MetadataPacketCount;
        Regex rgxAlphaNum;

        [NotNull]
        private readonly List<char> trialDataChars;

        private int trialDataSendOffset;

        [NotNull]
        private readonly StringBuilder builder;

        [NotNull]
        private readonly List<UnityWebRequest> activeRequests;

        public DataUploadHandler(string url, int floatPrecision = 5)
        {
            if (string.IsNullOrWhiteSpace(url))
            {
                throw new ArgumentNullException(url);
            }

            this.url = url;
            this.participantId = Prepare(Guid.Empty.ToString());
            this.sessionId = Prepare(Guid.Empty.ToString());
            this.expID = Prepare(Guid.Empty.ToString());
            this.trialIndex = -1;
            this.floatPrecision = $"F{floatPrecision}";

            this.trialDataChars = new List<char>();
            this.builder = new StringBuilder();
            this.activeRequests = new List<UnityWebRequest>();

            this.MetadataPacketCount = new Dictionary<string,int>();
            this.rgxAlphaNum = new Regex("[^a-zA-Z0-9]");

            CallProvider.AddUpdateListener(OnUpdate);
        }

        public void WriteUserData(string id,string sid,string expID, int age, Gender gender, int group, DateTime date)
        {
            this.participantId = Prepare(id);
            this.sessionId=Prepare(sid);
            this.expID=Prepare(expID);

            var data = new RawUserData
            {
                id = this.participantId,
                sid = this.sessionId,
                eid = this.expID,
                age = age.ToString(),
                ge = (gender == Gender.Male ? "m" : (gender == Gender.Female ? "f" : "o")),
                gr = group.ToString(),
                da = Prepare(date.ToShortDateString()) + ' ' + Prepare(date.ToShortTimeString())
            };

            var jsonData = JsonUtility.ToJson(data);

            Debug.Log("Sending user data.");
            Send(jsonData);
        }

        public void WriteTrialHeader(TrialData trialData)
        {
            trialIndex++;

            var data = new RawTrialHeader
            {
                id = this.participantId,
                sid = this.sessionId,
                tnum = (trialIndex).ToString(),
                td = trialData.TargetId.ToString(),
                tm = Prepare(trialData.TargetMaterialName),
                st = Prepare(trialData.StartTime.ToString(floatPrecision)),
            };

            trialPackageIndex = 0;
            trialStartTime = trialData.StartTime;
            trialDataChars.Clear();
            trialDataSendOffset = 0;

            var jsonData = JsonUtility.ToJson(data);

            Debug.Log("Sending trial header.");
            Send(jsonData);
        }

        public void WriteTrialData(TrackingEntry entry)
        {
            builder.Append(Prepare(entry.Time));
            builder.Append(',');
            builder.Append(Prepare(entry.Position.x));
            builder.Append(',');
            builder.Append(Prepare(entry.Position.y));
            builder.Append(',');
            builder.Append(Prepare(entry.Position.z));
            builder.Append(',');
            builder.Append(Prepare(entry.ViewAzimuth));
            builder.Append(',');
            builder.Append(Prepare(entry.ViewElevation));
            if (entry.ObservationSpace != null && entry.ObservationSpace.Length > 0)
            {
                builder.Append(',');
                for (int i = 0; i < entry.ObservationSpace.Length; i++)
                {
                    builder.Append(Prepare(entry.ObservationSpace[i]));
                    if (i < entry.ObservationSpace.Length - 1) builder.Append(';');
                }
            }
            builder.AppendLine();

            var chars = new char[builder.Length];
            for (var i = 0; i < builder.Length; i++)
            {
                chars[i] = builder[i];
            }

            var minCapacity = trialDataChars.Count + builder.Length;
            if (trialDataChars.Capacity < minCapacity)
            {
                trialDataChars.Capacity = minCapacity;
            }

            for (var i = 0; i < builder.Length; i++)
            {
                trialDataChars.Add(builder[i]);
            }

            builder.Clear();

            if (trialDataChars.Count - trialDataSendOffset >= TargetDataPackageChars)
            {
                Debug.Log($"Sending trial data. [Size: {trialDataChars.Count - trialDataSendOffset}]");
                SendData();
            }
        }

        public void WriteTrialTail(float endTime, float totalTrialTime, float totalTimeSinceStart)
        {
            Debug.Log($"Sending trial data. [Size: {trialDataChars.Count - trialDataSendOffset}]");
            SendData();

            var checksum = Md5Hash(trialDataChars.ToArray());

            var data = new RawDataPacketTail
            {
                id = this.participantId,
                sid = this.sessionId,
                pnum = this.trialPackageIndex.ToString(),
                tnum = trialIndex.ToString(),
                checksum = checksum,
                et = endTime.ToString(floatPrecision),
                tspan = (endTime - trialStartTime).ToString(floatPrecision),
                tsofar = totalTrialTime.ToString(floatPrecision),
                tsstart = totalTimeSinceStart.ToString(floatPrecision),
                //dbg = new string(trialDataChars.ToArray())
            };

            //trialPackageIndex++;

            var jsonData = JsonUtility.ToJson(data);

            Debug.Log("Sending trial tail.");
            Send(jsonData);
        }

        public void WriteMetaData(string category, string metaData)
        {
            category=this.rgxAlphaNum.Replace(category, "");
            if (this.MetadataPacketCount.ContainsKey(category)) { 
                this.MetadataPacketCount[category]+=1;
            }else{
                this.MetadataPacketCount[category]=0;
            }
            
            var data = new RawMetaData()
            {
                id = this.participantId,
                sid = this.sessionId,
                eid = this.expID,
                mid = this.MetadataPacketCount[category].ToString(),
                cat = category,
                meta = metaData,
                date = DateTime.Now.ToString(DateTimeFormat) 
            };

            //trialPackageIndex++;

            var jsonData = JsonUtility.ToJson(data);

            Debug.Log($"Sending meta data (cat: \"{category}\" data: \"{metaData}\")");
            Send(jsonData);
        }

        private void Send(string text)
        {
            var data = Encoding.UTF8.GetBytes(text);
            Post(data);
        }

        private void SendData()
        {
            if (trialDataSendOffset >= trialDataChars.Count)
            {
                return;
            }

            var minCapacity = trialDataChars.Count - trialDataSendOffset;
            builder.EnsureCapacity(minCapacity);

            for (var i = trialDataSendOffset; i < trialDataChars.Count; i++)
            {
                builder.Append(trialDataChars[i]);
            }

            var rawData = new RawTrialData();
            rawData.id = this.participantId;
            rawData.sid = this.sessionId;
            rawData.DATA = builder.ToString();
            rawData.pid = trialPackageIndex.ToString();
            rawData.tnum = trialIndex.ToString();

            trialPackageIndex++;
            trialDataSendOffset = trialDataChars.Count;
            builder.Clear();

            var json = JsonUtility.ToJson(rawData);
            var bytes = Encoding.UTF8.GetBytes(json);

            Post(bytes);
        }

        private void Post(byte[] data)
        {
            var request = new UnityWebRequest(url, "POST");
            request.uploadHandler = new UploadHandlerRaw(data);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.certificateHandler = new BypassCertificate();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SendWebRequest();

            activeRequests.Add(request);
            // TODO, does it need to yield "request.SendWebRequest()"
            // TODO Similar code code be reused from other parts
            if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.Log(request.error);
            }
        }

        private void OnUpdate()
        {
            for (var i = activeRequests.Count - 1; i >= 0; i--)
            {
                var request = activeRequests[i];

                if (request.result == UnityWebRequest.Result.ConnectionError)
                {
                    Debug.Log("Network error while sending: " + request.error);
                }
                else if (request.result == UnityWebRequest.Result.ProtocolError)
                {
                    Debug.Log("Http error while sending: " + request.error);
                }
                else if(request.isDone)
                {
                    Debug.Log("Request sent, received: " + request.downloadHandler.text);
                }
                else
                {
                    continue;
                }

                activeRequests.RemoveAt(i);
            }
        }

        private static string Prepare(float value)
        {
            return Prepare(value.ToString(CultureInfo.InvariantCulture));
        }

        private static string Prepare(string value)
        {
            return value.Replace(',', '.');
        }

        private string Md5Hash(char[] data)
        {
            var textBytes = Encoding.UTF8.GetBytes(data);
            var cryptoBytes = MD5.Create().ComputeHash(textBytes);

            foreach (var b in cryptoBytes)
            {
                builder.Append($"{b:x2}");
            }

            var result = builder.ToString();
            builder.Clear();

            return result;
        }

        [Serializable]
        struct RawUserData
        {
            public string id;
            public string sid;      
            public string eid;
            public string age;
            public string ge;
            public string gr;
            public string da;
        }

        [Serializable]
        struct RawTrialHeader
        {
            public string id;
            public string sid;
            public string tnum;
            public string td;
            public string tm;
            public string st; 
        }

        [Serializable]
        struct RawTrialData
        {
            public string id;
            public string sid;
            public string pid;
            public string tnum;
            public string DATA; 
        }

        [Serializable]
        struct RawDataPacketTail
        {
            public string id;

            public string sid;
            public string tnum;
            public string pnum;
            public string checksum;
            public string et;
            public string tspan;
            public string tsofar;
            public string tsstart; 
        }

        [Serializable]
        struct RawMetaData
        {
            public string id;
            public string sid;      
            public string eid;
            public string mid;
            public string cat;
            public string meta;
            public string date;
        }

        private class BypassCertificate : CertificateHandler
        {
            protected override bool ValidateCertificate(byte[] certificateData)
            {
                //Simply return true no matter what
                return true;
            }
        }

    }
}
