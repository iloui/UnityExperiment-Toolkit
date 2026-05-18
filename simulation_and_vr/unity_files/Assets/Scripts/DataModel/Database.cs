using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Assets.Scripts;
using Assets.Scripts.DataRecording;
using JetBrains.Annotations;
using UnityEngine;

/// <summary>
/// Stores data for a single participant but for the whole experiment
/// </summary>
public static class Database
{
    public static string ParticipantPC { get; set; }

    public static int ParticipantGroup { get; set; }

    public static string ParticipantId { get; set; }

    public static string ExperimentId { get; set; }

    public static string SessionId { get; set; }

    public static string DataCollectionServerURL { get; set; }

    public static int ParticipantAge { get; set; }

    public static Gender ParticipantGender { get; set; }

    public static event Action<int> NextTrialStarted; 

    public static readonly List<TrialData> TrialResults;
    private static TrialData currentTrial = new TrialData();

    private static bool sentUserData;

    public static TrialData CurrentTrial
    {
        get => currentTrial;
        set
        {
            if (currentTrial == value)
            {
                return;
            }

            if (currentTrial != null)
            {
                currentTrial.TrackingDataAdded -= CurrentTrialOnTrackingDataAdded;
            }

            currentTrial = value;

            if (currentTrial != null)
            {
                currentTrial.TrackingDataAdded += CurrentTrialOnTrackingDataAdded;
            }
        }
    }

    static Database()
    {       
        ParticipantAge = -1;
        TrialResults = new List<TrialData>();

        writer = new EmptyWriter();
    }

    public static float TotalTrialTime
    {
        get
        {
            return TrialResults.Sum(tr => tr.EndTime - tr.StartTime) + Time.time - CurrentTrial.StartTime;
        }
    }
     
    public static float TotalTimeSinceStart
    {
        get
        {
            if (TrialResults.Count == 0)
            {
                return Time.time - CurrentTrial.StartTime;
            }

            return Time.time - TrialResults.Min(tr => tr.StartTime);
        }
    }

    [CanBeNull]
    private static IDataWriter writer;

    private static DataUploadHandler uploadHandler;

    public static void StartNewTrial(int targetId, string targetMaterialName)
    {
        CurrentTrial = new TrialData();
        CurrentTrial.TargetId = targetId;
        CurrentTrial.TargetMaterialName = targetMaterialName;
        CurrentTrial.StartTime = Time.time;

        if (uploadHandler == null)
        {
            uploadHandler = new DataUploadHandler(DataCollectionServerURL);
            uploadHandler.WriteUserData(ParticipantId ?? "NonePID", SessionId ?? "NoneSID",ExperimentId ?? "NoneEID", ParticipantAge, ParticipantGender, ParticipantGroup, DateTime.Now);
        }

        if (!sentUserData)
        {
            SendParticipantInfo();
        }

        WriteHeader();
        uploadHandler.WriteTrialHeader(CurrentTrial);

        OnNextTrialStarted(targetId);
    }

    public static void EndTrial()
    {
        if (CurrentTrial.TargetId != -1)
        {
            CurrentTrial.EndTime = Time.time;
            TrialResults.Add(CurrentTrial);
            CurrentTrial = new TrialData();

            uploadHandler.WriteTrialTail(Time.time, TotalTrialTime, TotalTimeSinceStart);
        }
    }

    public static void SendParticipantInfo()
    {
        if (uploadHandler == null)
        {
            uploadHandler = new DataUploadHandler(DataCollectionServerURL);
        }

        uploadHandler.WriteUserData(ParticipantId ?? "NonePID", SessionId ?? "NoneSID",ExperimentId ?? "NoneEID", ParticipantAge, ParticipantGender, ParticipantGroup, DateTime.Now);
        sentUserData = true;
    }

    public static void SendMetaData(string category, string metaData)
    {
        if (uploadHandler == null)
        {
            uploadHandler = new DataUploadHandler(DataCollectionServerURL);
        }

        uploadHandler.WriteMetaData(category, metaData);
    }

    private static void CurrentTrialOnTrackingDataAdded(TrackingEntry entry)
    {
        WriteTrackingEntry(entry);

        uploadHandler.WriteTrialData(entry);
    }
     
    /// <summary>
    /// Gets an unique path in order to make sure no existing file is overwritten
    /// </summary>
    private static string GetOutputPath()
    {
        var path = Path.Combine(GetOutputFolder(), $"Participant_PC_{ParticipantPC ?? "[None]"}@Participant_{ParticipantId ?? "[None]"}@Group_{ParticipantGroup}@Date_{DateTime.Now:dd-MM-yyyy}@Time_{DateTime.Now:hh-mm}.csv");
        var altPath = path;
        var altIndex = 2;

        while (File.Exists(altPath))
        {
            altPath = path + $" ({altIndex})";
            altIndex++;
        }

        return altPath;
    }

    /// <summary>
    /// Gets the output folder.
    /// </summary>
    /// <returns>System.String.</returns>
    private static string GetOutputFolder()
    {
        var di = new DirectoryInfo(Application.dataPath);
        if (di.Exists)
        {
            if (!di.Attributes.HasFlag(FileAttributes.ReadOnly))
            {
                var path = Path.Combine(Application.dataPath, "Trial Results");
                if (!Directory.Exists(path))
                {
                    Directory.CreateDirectory(path);
                }

                return path;
            }
        }

        return Application.persistentDataPath;
    }

    private static void WriteHeader()
    {
        if (writer == null)
        {
            return;
        }

        writer.Write("Participant id:,");
        writer.WriteLine(ParticipantId ?? "[None]");

        writer.Write("Experiment id:,");
        writer.WriteLine(ExperimentId ?? "[None]");

        writer.Write("Participant age:,");
        writer.WriteLine(ParticipantAge.ToString());

        writer.Write("Participant gender:,");
        writer.WriteLine(ParticipantGender.ToString());

        writer.Write("Group:,");
        writer.WriteLine(ParticipantGroup.ToString());

        writer.Write("Date:,");
        writer.WriteLine(Prepare(DateTime.Now.ToShortDateString()) + ' ' + Prepare(DateTime.Now.ToShortTimeString()));

        writer.WriteLine();
        writer.WriteLine();
    }

    private static void WriteTrialHeader(TrialData data, int index)
    {
        if (writer == null)
        {
            return;
        }

        writer.WriteLine();
        writer.WriteLine();

        writer.Write("Trial no.:,");
        writer.WriteLine((index + 1).ToString());

        writer.Write("Target id:,");
        writer.WriteLine(data.TargetId.ToString());

        writer.Write("Target material:,");
        writer.WriteLine(data.TargetMaterialName);

        writer.Write("Start time:,");
        writer.WriteLine(data.StartTime);

        writer.Write("End time:,");
        writer.WriteLine(data.EndTime);

        writer.Write("Time span:,");
        writer.WriteLine(data.EndTime - data.StartTime);

        writer.Write("Total trial time so far:,");
        writer.WriteLine(Database.TotalTrialTime);

        writer.Write("Total time since start:,");
        writer.WriteLine(Database.TotalTimeSinceStart);

        writer.WriteLine();
        writer.WriteLine("Time,X,Y,Z,View azimuth, View elevation");
    }

    private static void WriteTrackingEntry(TrackingEntry entry)
    {
        if (writer == null)
        {
            return;
        }

        writer.Write(Prepare(entry.Time));
        writer.Write(',');
        writer.Write(Prepare(entry.Position.x));
        writer.Write(',');
        writer.Write(Prepare(entry.Position.y));
        writer.Write(',');
        writer.Write(Prepare(entry.Position.z));
        writer.Write(',');
        writer.Write(Prepare(entry.ViewAzimuth));
        writer.Write(',');
        writer.Write(Prepare(entry.ViewElevation));
        writer.WriteLine();
    }

    private static string Prepare(float value)
    {
        return Prepare(value.ToString(CultureInfo.InvariantCulture));
    }

    private static string Prepare(string value)
    {
        return value.Replace(',', '.');
    }

    private static void OnNextTrialStarted(int targetDoorNumber)
    {
        NextTrialStarted?.Invoke(targetDoorNumber);
    }
}
