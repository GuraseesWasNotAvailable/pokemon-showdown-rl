// PSBot launcher.
//
// Opens Pokemon Showdown in a dedicated Chrome window and loads psbot.js into it, so the bot
// runs without pasting anything into the console. It stays in the background while that window
// is open and looks after an unattended run:
//   - puts the bot back after a page reload, and loads a changed psbot.js into the open page
//     without reloading it
//   - writes what the bot hands over to disk: finished games, a status heartbeat, the log
//   - keeps the computer awake while games are running
//   - reloads a page that has stopped answering, reopens a closed tab, restarts a crashed Chrome
// It exits when the window is closed.
//
// Build: build-launcher.cmd (uses the C# compiler that ships with Windows).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Net.WebSockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Microsoft.Win32;

static class Program
{
	// PSBOT_SITE, PSBOT_DATA_DIR and PSBOT_CHROME_ARGS exist for testing against a local server
	// with a separate browser profile. Nobody needs them to play.
	static string Site = "https://play.pokemonshowdown.com/";
	static string ExtraArgs = "";
	const string Title = "PSBot";

	static readonly JavaScriptSerializer Json = new JavaScriptSerializer { MaxJsonLength = 16 * 1024 * 1024 };
	static readonly Dictionary<string, ClientWebSocket> Attached = new Dictionary<string, ClientWebSocket>();
	// The id Chrome gave the script registered for new documents, per tab, so it can be replaced.
	static readonly Dictionary<string, string> ScriptIds = new Dictionary<string, string>();
	static readonly object FileLock = new object();
	static string logFile;
	static string wrapper;
	static string botSource;
	static string botStamp = "";
	static string exeDir;
	static string battlesDir;
	static string logsDir;
	static readonly HashSet<string> Indexed = new HashSet<string>();

	// What the page last reported about itself.
	static DateTime lastBeat = DateTime.MinValue;
	static bool lastAuto;
	static int lastLive;
	static DateTime lastNudge = DateTime.MinValue;
	static DateTime lastTabOpen = DateTime.MinValue;
	static readonly List<DateTime> Relaunches = new List<DateTime>();
	static Process browserProc;
	static string browserPath;
	static string profilePath;

	[DllImport("user32.dll")]
	static extern bool AllowSetForegroundWindow(int processId);

	[DllImport("kernel32.dll")]
	static extern uint SetThreadExecutionState(uint flags);
	const uint ES_SYSTEM_REQUIRED = 0x00000001;

	[STAThread]
	static int Main()
	{
		try
		{
			return Run();
		}
		catch (Exception e)
		{
			Log("fatal: " + e);
			MessageBox.Show(e.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
			return 1;
		}
	}

	static int Run()
	{
		string dataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PSBot");
		string mutexName = "PSBotLauncher.SingleInstance";
		string envDir = Environment.GetEnvironmentVariable("PSBOT_DATA_DIR");
		if (!string.IsNullOrEmpty(envDir))
		{
			dataDir = envDir;
			mutexName += "." + Regex.Replace(envDir.ToLowerInvariant(), "[^a-z0-9]", "");
			if (mutexName.Length > 200) mutexName = mutexName.Substring(0, 200);
		}
		string envSite = Environment.GetEnvironmentVariable("PSBOT_SITE");
		if (!string.IsNullOrEmpty(envSite)) Site = envSite;
		ExtraArgs = Environment.GetEnvironmentVariable("PSBOT_CHROME_ARGS") ?? "";
		// Recorded games, the status file and the logs go next to the exe, where they are easy to
		// find and to analyse later.
		exeDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
		battlesDir = Path.Combine(exeDir, "battles");
		logsDir = Path.Combine(exeDir, "logs");
		// Chrome only allows remote control of a profile kept outside its default folder, so the
		// bot gets its own. It holds nothing but the Showdown login.
		string profileDir = Path.Combine(dataDir, "chrome-profile");
		Directory.CreateDirectory(profileDir);
		logFile = Path.Combine(dataDir, "launcher.log");
		string portFile = Path.Combine(dataDir, "port.txt");

		string browser = FindBrowser();
		if (browser == null) throw new Exception("Google Chrome was not found on this computer.\n\nInstall Chrome and run PSBot again.");
		browserPath = browser;
		profilePath = profileDir;
		botSource = LoadBotSource();
		botStamp = BotStamp();
		wrapper = BuildWrapper(botSource);

		using (Mutex mutex = new Mutex(false, mutexName))
		{
			bool owner = Acquire(mutex, 0);
			try
			{
				int port = ReadPort(portFile);
				bool alive = port > 0 && IsAlive(port);
				if (!alive && !owner)
				{
					// Another copy holds the lock but has no window: it is either still starting up
					// or shutting down after its window was closed. Wait for it to finish.
					owner = Acquire(mutex, 20000);
					port = ReadPort(portFile);
					alive = port > 0 && IsAlive(port);
					if (!alive && !owner) return 0;
				}
				if (alive)
				{
					// The bot window is already open. Bring its Showdown tab forward rather than
					// opening a second one, which would put two bots on the same battle.
					try { AllowSetForegroundWindow(-1); } catch (Exception) { }
					if (!FocusShowdownTab(port)) StartBrowser(browser, profileDir, port);
					if (!owner) return 0;
					Log("rejoined the open bot window on port " + port);
				}
				else
				{
					port = FreePort();
					File.WriteAllText(portFile, port.ToString());
					try { if (File.Exists(logFile) && new FileInfo(logFile).Length > 2 * 1024 * 1024) File.WriteAllText(logFile, ""); } catch (Exception) { }
					Log("starting " + browser + " on port " + port);
					StartBrowser(browser, profileDir, port);
					if (!WaitAlive(port, 30000))
					{
						throw new Exception("Chrome opened, but PSBot could not connect to it.\n\nClose every PSBot Chrome window and run PSBot again.");
					}
				}
				AttachLoop(port);
			}
			finally
			{
				if (owner) { try { mutex.ReleaseMutex(); } catch (Exception) { } }
			}
		}
		return 0;
	}

	// ------------------------------------------------------------------ browser

	static string FindBrowser()
	{
		List<string> candidates = new List<string>();
		foreach (RegistryKey root in new RegistryKey[] { Registry.CurrentUser, Registry.LocalMachine })
		{
			try
			{
				using (RegistryKey k = root.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"))
				{
					if (k == null) continue;
					string v = k.GetValue(null) as string;
					if (!string.IsNullOrEmpty(v)) candidates.Add(v.Trim('"'));
				}
			}
			catch (Exception) { }
		}
		string pf = Environment.GetEnvironmentVariable("ProgramFiles");
		string pf86 = Environment.GetEnvironmentVariable("ProgramFiles(x86)");
		string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
		foreach (string b in new string[] { pf, pf86, local })
		{
			if (!string.IsNullOrEmpty(b)) candidates.Add(Path.Combine(b, @"Google\Chrome\Application\chrome.exe"));
		}
		// Edge takes the same switches, so it works as a stand-in when Chrome is missing.
		foreach (string b in new string[] { pf86, pf })
		{
			if (!string.IsNullOrEmpty(b)) candidates.Add(Path.Combine(b, @"Microsoft\Edge\Application\msedge.exe"));
		}
		foreach (string p in candidates)
		{
			if (File.Exists(p)) return p;
		}
		return null;
	}

	static void StartBrowser(string browser, string profileDir, int port)
	{
		// The three "disable" switches stop Chrome from slowing the page down when its window is
		// covered or the screen is off, which is exactly when an unattended run is happening.
		string args = "--user-data-dir=\"" + profileDir + "\" --remote-debugging-port=" + port +
			" --no-first-run --no-default-browser-check --disable-background-timer-throttling" +
			" --disable-backgrounding-occluded-windows --disable-renderer-backgrounding " + (ExtraArgs.Length > 0 ? ExtraArgs + " " : "") + "--new-window " + Site;
		ProcessStartInfo psi = new ProcessStartInfo(browser, args);
		psi.UseShellExecute = false;
		browserProc = Process.Start(psi);
	}

	static bool Acquire(Mutex mutex, int waitMs)
	{
		try
		{
			return mutex.WaitOne(waitMs);
		}
		catch (AbandonedMutexException)
		{
			return true; // the previous owner was killed; the lock is ours now
		}
	}

	static int FreePort()
	{
		TcpListener l = new TcpListener(IPAddress.Loopback, 0);
		l.Start();
		int port = ((IPEndPoint)l.LocalEndpoint).Port;
		l.Stop();
		return port;
	}

	static int ReadPort(string portFile)
	{
		try
		{
			int port;
			if (File.Exists(portFile) && int.TryParse(File.ReadAllText(portFile).Trim(), out port)) return port;
		}
		catch (Exception) { }
		return 0;
	}

	static string Base(int port)
	{
		return "http://127.0.0.1:" + port;
	}

	static string Http(string url, int timeoutMs)
	{
		HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
		req.Method = "GET";
		req.Proxy = null;
		req.KeepAlive = false;
		req.Timeout = timeoutMs;
		req.ReadWriteTimeout = timeoutMs;
		using (WebResponse resp = req.GetResponse())
		using (StreamReader r = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
		{
			return r.ReadToEnd();
		}
	}

	static bool IsAlive(int port)
	{
		try
		{
			return Http(Base(port) + "/json/version", 1500).IndexOf("webSocketDebuggerUrl", StringComparison.Ordinal) >= 0;
		}
		catch (Exception)
		{
			return false;
		}
	}

	static bool WaitAlive(int port, int timeoutMs)
	{
		Stopwatch sw = Stopwatch.StartNew();
		while (sw.ElapsedMilliseconds < timeoutMs)
		{
			if (IsAlive(port)) return true;
			Thread.Sleep(400);
		}
		return false;
	}

	static List<Dictionary<string, object>> Pages(int port)
	{
		List<Dictionary<string, object>> pages = new List<Dictionary<string, object>>();
		object[] targets = Json.DeserializeObject(Http(Base(port) + "/json/list", 3000)) as object[];
		if (targets == null) return pages;
		foreach (object o in targets)
		{
			Dictionary<string, object> t = o as Dictionary<string, object>;
			if (t != null && Str(t, "type") == "page" && IsShowdown(Str(t, "url"))) pages.Add(t);
		}
		return pages;
	}

	static string Str(Dictionary<string, object> d, string key)
	{
		object v;
		return d.TryGetValue(key, out v) && v != null ? v.ToString() : "";
	}

	static bool IsShowdown(string url)
	{
		Uri u;
		if (!Uri.TryCreate(url, UriKind.Absolute, out u)) return false;
		string h = u.Host.ToLowerInvariant();
		return h == "play.pokemonshowdown.com" || h.EndsWith(".psim.us");
	}

	static bool FocusShowdownTab(int port)
	{
		try
		{
			List<Dictionary<string, object>> pages = Pages(port);
			if (pages.Count == 0) return false;
			Http(Base(port) + "/json/activate/" + Str(pages[0], "id"), 3000);
			return true;
		}
		catch (Exception)
		{
			return false;
		}
	}

	// ------------------------------------------------------------------ injection

	static string LoadBotSource()
	{
		Assembly asm = Assembly.GetExecutingAssembly();
		// A psbot.js next to the exe wins, so the bot can be updated without rebuilding.
		string beside = Path.Combine(Path.GetDirectoryName(asm.Location), "psbot.js");
		if (File.Exists(beside)) return File.ReadAllText(beside, Encoding.UTF8);
		using (Stream s = asm.GetManifestResourceStream("psbot.js"))
		{
			if (s == null) throw new Exception("psbot.js is missing.\n\nKeep it in the same folder as PSBot.exe.");
			using (StreamReader r = new StreamReader(s, Encoding.UTF8))
			{
				return r.ReadToEnd();
			}
		}
	}

	// Size and time of the psbot.js next to the exe: changes when a new version is dropped in.
	static string BotStamp()
	{
		try
		{
			FileInfo f = new FileInfo(Path.Combine(exeDir, "psbot.js"));
			return f.Exists ? f.Length + "@" + f.LastWriteTimeUtc.Ticks : "";
		}
		catch (Exception)
		{
			return "";
		}
	}

	// Run in a page that already has the bot: load the new source in place. The new copy takes
	// over the running games from the old one. If it fails to load, the old one is put back.
	static string BuildSwap(string src)
	{
		const string head = @"(function () {
	var src = ";
		const string tail = @";
	if (!window.PSBot) return 'no bot in this page';
	try { new Function(src); } catch (e) { return 'rejected, syntax error: ' + e.message; }
	var old = window.__psbotSrc || null;
	// A copy too old to hand its state over still gets its switches carried across.
	var wasOn = false, wasAuto = false;
	try { wasOn = !!window.PSBot.config.enabled; wasAuto = !!window.PSBot.config.autoSearch; } catch (e0) {}
	try {
		(0, eval)(src);
		window.__psbotSrc = src;
		if (wasOn && window.PSBot && !window.PSBot.config.enabled) window.PSBot.start(wasAuto ? {autoSearch: true} : undefined);
		return 'ok ' + (window.PSBot ? window.PSBot.version + ' ' + (window.PSBot.build || '') : '?');
	} catch (e) {
		var msg = 'failed: ' + e.message;
		try { if (old) { (0, eval)(old); msg += ' (previous version restored)'; } } catch (e2) { msg += ' (restore failed: ' + e2.message + ')'; }
		return msg;
	}
})()";
		return head + Json.Serialize(src) + tail;
	}

	// Runs in every new document of an attached tab. It waits for the Showdown client to finish
	// loading, evaluates the bot, and switches it on unless another tab already has it running.
	static string BuildWrapper(string botSource)
	{
		const string head = @"(function () {
	if (window.top !== window) return;
	var h = location.hostname;
	if (h !== 'play.pokemonshowdown.com' && !/\.psim\.us$/.test(h)) return;
	if (window.__psbotBoot) return;
	window.__psbotBoot = true;
	var src = ";
		const string tail = @";
	var tries = 0;
	var timer = setInterval(function () {
		tries++;
		if (!(window.app && window.app.user && window.BattleMovedex && document.body)) {
			if (tries > 480) clearInterval(timer);
			return;
		}
		clearInterval(timer);
		try { (0, eval)(src); window.__psbotSrc = src; } catch (e) { console.error('[PSBot] failed to load', e); return; }
		var othersOn = false;
		try {
			var ch = new BroadcastChannel('psbot');
			ch.onmessage = function (ev) {
				if (ev.data === 'who') { if (window.PSBot && window.PSBot.config.enabled) ch.postMessage('on'); }
				else if (ev.data === 'on') othersOn = true;
			};
			ch.postMessage('who');
		} catch (e) {}
		setTimeout(function () {
			if (!othersOn && window.PSBot && !window.PSBot.config.enabled) window.PSBot.start();
		}, 700);
	}, 250);
})();";
		return head + Json.Serialize(botSource) + tail;
	}

	static void AttachLoop(int port)
	{
		int fails = 0;
		string pendingStamp = null;
		while (true)
		{
			List<Dictionary<string, object>> pages = null;
			try
			{
				pages = Pages(port);
				fails = 0;
			}
			catch (Exception)
			{
				fails++;
			}
			if (fails >= 3)
			{
				// The browser is gone. A window the user closed means stop; a crash in the middle
				// of an unattended run means start it again.
				if (!ShouldRelaunch())
				{
					Log("browser closed, exiting");
					return;
				}
				Relaunches.Add(DateTime.UtcNow);
				Note("launcher-relaunch", "Chrome went away during a run, starting it again");
				lock (Attached) { Attached.Clear(); }
				lock (ScriptIds) { ScriptIds.Clear(); }
				try
				{
					StartBrowser(browserPath, profilePath, port);
					if (!WaitAlive(port, 45000)) { Log("could not restart Chrome, exiting"); return; }
				}
				catch (Exception e)
				{
					Log("relaunch failed: " + e.Message);
					return;
				}
				lastBeat = DateTime.UtcNow;
				fails = 0;
				continue;
			}
			if (fails > 0)
			{
				Thread.Sleep(400);
				continue;
			}
			if (pages != null)
			{
				foreach (Dictionary<string, object> t in pages)
				{
					string id = Str(t, "id");
					string ws = Str(t, "webSocketDebuggerUrl");
					if (id.Length == 0 || ws.Length == 0) continue;
					lock (Attached)
					{
						if (Attached.ContainsKey(id)) continue;
					}
					try
					{
						Attach(id, ws);
					}
					catch (Exception e)
					{
						Log("attach failed for " + id + ": " + e.Message);
					}
				}
			}

			bool busy = lastAuto || lastLive > 0;
			bool fresh = (DateTime.UtcNow - lastBeat).TotalSeconds < 120;
			// Games are running: keep the computer from going to sleep.
			if (busy && fresh) { try { SetThreadExecutionState(ES_SYSTEM_REQUIRED); } catch (Exception) { } }

			// A new psbot.js next to the exe: wait until it has stopped changing, then load it.
			string stamp = BotStamp();
			if (stamp.Length > 0 && stamp != botStamp)
			{
				if (stamp == pendingStamp)
				{
					pendingStamp = null;
					botStamp = stamp;
					try { HotLoad(); } catch (Exception e) { Note("launcher-hotload", "failed: " + e.Message); }
				}
				else
				{
					pendingStamp = stamp;
				}
			}

			// The run was on and the page has gone quiet: bring it back.
			if (lastAuto && lastBeat != DateTime.MinValue)
			{
				double quiet = (DateTime.UtcNow - lastBeat).TotalSeconds;
				if (pages != null && pages.Count == 0 && quiet > 45 && (DateTime.UtcNow - lastTabOpen).TotalSeconds > 120)
				{
					lastTabOpen = DateTime.UtcNow;
					Note("launcher-tab", "the Showdown tab is gone, opening it again");
					try { OpenTab(port); } catch (Exception e) { Log("could not open a tab: " + e.Message); }
				}
				else if (pages != null && pages.Count > 0 && quiet > 150 && (DateTime.UtcNow - lastNudge).TotalSeconds > 180)
				{
					lastNudge = DateTime.UtcNow;
					Note("launcher-reload", "no heartbeat from the page for " + (int)quiet + "s, reloading it");
					Navigate();
				}
			}
			Thread.Sleep(1500);
		}
	}

	static bool ShouldRelaunch()
	{
		if (!lastAuto || lastBeat == DateTime.MinValue) return false;
		if ((DateTime.UtcNow - lastBeat).TotalSeconds > 300) return false;
		if (File.Exists(Path.Combine(exeDir, "STOP"))) return false;
		Relaunches.RemoveAll(delegate (DateTime t) { return (DateTime.UtcNow - t).TotalMinutes > 60; });
		if (Relaunches.Count >= 6) return false;
		// A window closed by hand ends Chrome with exit code 0. Anything else was not the user.
		try
		{
			if (browserProc != null && browserProc.HasExited && browserProc.ExitCode == 0) return false;
		}
		catch (Exception) { }
		return browserProc != null;
	}

	static void OpenTab(int port)
	{
		HttpWebRequest req = (HttpWebRequest)WebRequest.Create(Base(port) + "/json/new?" + Site);
		req.Method = "PUT";
		req.Proxy = null;
		req.ContentLength = 0;
		req.Timeout = 5000;
		using (WebResponse resp = req.GetResponse()) { }
	}

	static List<KeyValuePair<string, ClientWebSocket>> Sockets()
	{
		lock (Attached)
		{
			return new List<KeyValuePair<string, ClientWebSocket>>(Attached);
		}
	}

	// Send every attached tab back to Showdown. Covers a frozen page and a page stuck on a
	// "no internet" error alike.
	static void Navigate()
	{
		foreach (KeyValuePair<string, ClientWebSocket> kv in Sockets())
		{
			try
			{
				Dictionary<string, object> p = new Dictionary<string, object>();
				p["url"] = Site;
				Send(kv.Value, 9, "Page.navigate", p);
			}
			catch (Exception e)
			{
				Log("navigate failed for " + kv.Key + ": " + e.Message);
			}
		}
	}

	// Swap the bot in every attached tab for the psbot.js now on disk, and register it for
	// future page loads in place of the old one.
	static void HotLoad()
	{
		string src = File.ReadAllText(Path.Combine(exeDir, "psbot.js"), Encoding.UTF8);
		if (src.Length < 1000) { Note("launcher-hotload", "psbot.js looks empty, keeping the running version"); return; }
		botSource = src;
		wrapper = BuildWrapper(src);
		string swap = BuildSwap(src);
		int n = 0;
		foreach (KeyValuePair<string, ClientWebSocket> kv in Sockets())
		{
			try
			{
				string oldId;
				lock (ScriptIds) { ScriptIds.TryGetValue(kv.Key, out oldId); }
				if (!string.IsNullOrEmpty(oldId))
				{
					Dictionary<string, object> rm = new Dictionary<string, object>();
					rm["identifier"] = oldId;
					Send(kv.Value, 6, "Page.removeScriptToEvaluateOnNewDocument", rm);
				}
				Dictionary<string, object> onNew = new Dictionary<string, object>();
				onNew["source"] = wrapper;
				Send(kv.Value, 2, "Page.addScriptToEvaluateOnNewDocument", onNew);
				Dictionary<string, object> now = new Dictionary<string, object>();
				now["expression"] = swap;
				now["returnByValue"] = true;
				Send(kv.Value, 7, "Runtime.evaluate", now);
				n++;
			}
			catch (Exception e)
			{
				Log("hot load failed for " + kv.Key + ": " + e.Message);
			}
		}
		Note("launcher-hotload", "sent the new psbot.js (" + src.Length + " chars) to " + n + " tab" + (n == 1 ? "" : "s"));
	}

	static void Attach(string id, string wsUrl)
	{
		ClientWebSocket ws = new ClientWebSocket();
		ws.Options.KeepAliveInterval = TimeSpan.Zero;
		if (!ws.ConnectAsync(new Uri(wsUrl), CancellationToken.None).Wait(8000)) throw new Exception("timed out connecting");
		lock (Attached)
		{
			Attached[id] = ws;
		}
		Thread reader = new Thread(delegate () { Drain(id, ws); });
		reader.IsBackground = true;
		reader.Start();
		try
		{
			Send(ws, 1, "Page.enable", null);
			// psbotSave(json) in the page arrives here as Runtime.bindingCalled: the bot uses it
			// to hand over each finished game for saving.
			Send(ws, 4, "Runtime.enable", null);
			Dictionary<string, object> binding = new Dictionary<string, object>();
			binding["name"] = "psbotSave";
			Send(ws, 5, "Runtime.addBinding", binding);
			// Registered for future documents, so the bot comes back after a reload.
			Dictionary<string, object> onNew = new Dictionary<string, object>();
			onNew["source"] = wrapper;
			Send(ws, 2, "Page.addScriptToEvaluateOnNewDocument", onNew);
			// And once now, for the page that is already loaded.
			Dictionary<string, object> now = new Dictionary<string, object>();
			now["expression"] = wrapper;
			Send(ws, 3, "Runtime.evaluate", now);
			Log("attached " + id);
		}
		catch (Exception)
		{
			try { ws.Abort(); } catch (Exception) { }
			throw;
		}
	}

	static void Send(ClientWebSocket ws, int id, string method, Dictionary<string, object> prms)
	{
		Dictionary<string, object> msg = new Dictionary<string, object>();
		msg["id"] = id;
		msg["method"] = method;
		if (prms != null) msg["params"] = prms;
		byte[] bytes = Encoding.UTF8.GetBytes(Json.Serialize(msg));
		if (!ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None).Wait(10000))
		{
			throw new Exception("timed out sending " + method);
		}
	}

	// Reads whatever the tab sends, saves the games the bot hands over, and notices when the tab
	// goes away. Everything else is discarded.
	static void Drain(string id, ClientWebSocket ws)
	{
		byte[] buf = new byte[65536];
		MemoryStream msg = new MemoryStream();
		try
		{
			while (ws.State == WebSocketState.Open)
			{
				WebSocketReceiveResult r = ws.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None).Result;
				if (r.MessageType == WebSocketMessageType.Close) break;
				if (msg.Length < 32 * 1024 * 1024) msg.Write(buf, 0, r.Count);
				if (!r.EndOfMessage) continue;
				if (msg.Length < 32 * 1024 * 1024)
				{
					string text = Encoding.UTF8.GetString(msg.GetBuffer(), 0, (int)msg.Length);
					if (text.IndexOf("\"Runtime.bindingCalled\"", StringComparison.Ordinal) >= 0) HandleBinding(text);
					else if (text.Length < 4096 && text.StartsWith("{\"id\":", StringComparison.Ordinal)) HandleReply(id, text);
				}
				msg.SetLength(0);
			}
		}
		catch (Exception) { }
		lock (Attached)
		{
			ClientWebSocket cur;
			if (Attached.TryGetValue(id, out cur) && cur == ws) Attached.Remove(id);
		}
		lock (ScriptIds) { ScriptIds.Remove(id); }
		try { ws.Dispose(); } catch (Exception) { }
		Log("detached " + id);
	}

	// Answers to the commands this program sent: the id of the registered script (needed to
	// replace it later) and the outcome of a hot load.
	static void HandleReply(string tabId, string text)
	{
		try
		{
			Dictionary<string, object> m = Json.DeserializeObject(text) as Dictionary<string, object>;
			if (m == null || !m.ContainsKey("id") || !m.ContainsKey("result")) return;
			int rid = Convert.ToInt32(m["id"]);
			Dictionary<string, object> res = m["result"] as Dictionary<string, object>;
			if (res == null) return;
			if (rid == 2)
			{
				string ident = Str(res, "identifier");
				if (ident.Length > 0) lock (ScriptIds) { ScriptIds[tabId] = ident; }
			}
			else if (rid == 7)
			{
				Dictionary<string, object> inner = res.ContainsKey("result") ? res["result"] as Dictionary<string, object> : null;
				string outcome = inner != null ? Str(inner, "value") : "";
				if (res.ContainsKey("exceptionDetails")) outcome = "threw before it could load";
				Note("launcher-hotload", "tab " + tabId.Substring(0, Math.Min(8, tabId.Length)) + ": " + outcome);
				Dictionary<string, object> rl = new Dictionary<string, object>();
				rl["t"] = DateTime.UtcNow.ToString("o");
				rl["outcome"] = outcome;
				rl["ok"] = outcome.StartsWith("ok ", StringComparison.Ordinal);
				WriteAtomic(Path.Combine(exeDir, "reload.json"), Json.Serialize(rl));
			}
		}
		catch (Exception) { }
	}

	static void HandleBinding(string text)
	{
		try
		{
			Dictionary<string, object> m = Json.DeserializeObject(text) as Dictionary<string, object>;
			if (m == null || Str(m, "method") != "Runtime.bindingCalled") return;
			Dictionary<string, object> p = m["params"] as Dictionary<string, object>;
			if (p == null || Str(p, "name") != "psbotSave") return;
			Dictionary<string, object> d = Json.DeserializeObject(Str(p, "payload")) as Dictionary<string, object>;
			if (d == null) return;
			string kind = Str(d, "kind");
			if (kind == "status") SaveStatus(Str(d, "body"));
			else if (kind == "log") SaveLog(Str(d, "day"), Str(d, "text"));
			else if (kind == "event") SaveEvent(Str(d, "line"));
			else SaveBattle(Str(d, "id"), Str(d, "day"), Str(d, "summary"), Str(d, "body"));
		}
		catch (Exception e)
		{
			Log("save failed: " + e.Message);
		}
	}

	// battles\<day>\<battle id>.json holds the full record; battles\index.jsonl has one summary
	// line per game. The request comes from a web page, so nothing in it is trusted as a path.
	static void SaveBattle(string id, string day, string summary, string body)
	{
		if (!Regex.IsMatch(id, "^[a-z0-9-]{1,100}$") || !Regex.IsMatch(day, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$")) return;
		if (body.Length == 0 || body.Length > 8000000) return;
		if (summary.Length == 0 || summary.Length > 20000 || summary.IndexOf('\n') >= 0 || summary.IndexOf('\r') >= 0) return;
		string dir = Path.Combine(battlesDir, day);
		Directory.CreateDirectory(dir);
		UTF8Encoding utf8 = new UTF8Encoding(false);
		File.WriteAllText(Path.Combine(dir, id + ".json"), body, utf8);
		lock (Indexed)
		{
			if (Indexed.Add(id)) File.AppendAllText(Path.Combine(battlesDir, "index.jsonl"), summary + "\n", utf8);
		}
		Log("saved " + id);
	}

	// status.json: what the bot is doing right now. Rewritten every half minute, so anything
	// watching the run can tell at a glance whether it is alive.
	static void SaveStatus(string body)
	{
		if (body.Length == 0 || body.Length > 100000) return;
		Dictionary<string, object> st = Json.DeserializeObject(body) as Dictionary<string, object>;
		if (st == null) return;
		lastBeat = DateTime.UtcNow;
		object v;
		lastAuto = st.TryGetValue("auto", out v) && v is bool && (bool)v;
		lastLive = st.TryGetValue("live", out v) && v != null ? Convert.ToInt32(v) : 0;
		WriteAtomic(Path.Combine(exeDir, "status.json"), body);
	}

	static void SaveLog(string day, string text)
	{
		if (!Regex.IsMatch(day, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$") || text.Length == 0 || text.Length > 2000000) return;
		lock (FileLock)
		{
			Directory.CreateDirectory(logsDir);
			File.AppendAllText(Path.Combine(logsDir, "bot-" + day + ".log"), text.Replace("\r", "") + "\n", new UTF8Encoding(false));
		}
	}

	static void SaveEvent(string line)
	{
		if (line.Length == 0 || line.Length > 20000 || line.IndexOf('\n') >= 0 || line.IndexOf('\r') >= 0) return;
		lock (FileLock)
		{
			Directory.CreateDirectory(logsDir);
			File.AppendAllText(Path.Combine(logsDir, "events.jsonl"), line + "\n", new UTF8Encoding(false));
		}
	}

	// Something this program did on its own, written where the bot's own events go.
	static void Note(string type, string detail)
	{
		Log(type + ": " + detail);
		try
		{
			Dictionary<string, object> ev = new Dictionary<string, object>();
			ev["t"] = DateTime.UtcNow.ToString("o");
			ev["type"] = type;
			ev["detail"] = detail;
			SaveEvent(Json.Serialize(ev));
		}
		catch (Exception) { }
	}

	static void WriteAtomic(string path, string text)
	{
		lock (FileLock)
		{
			try
			{
				string tmp = path + ".tmp";
				File.WriteAllText(tmp, text, new UTF8Encoding(false));
				if (File.Exists(path)) File.Replace(tmp, path, null); else File.Move(tmp, path);
			}
			catch (Exception)
			{
				try { File.WriteAllText(path, text, new UTF8Encoding(false)); } catch (Exception) { }
			}
		}
	}

	static void Log(string line)
	{
		if (logFile == null) return;
		try
		{
			File.AppendAllText(logFile, DateTime.Now.ToString("HH:mm:ss") + " " + line + Environment.NewLine);
		}
		catch (Exception) { }
	}
}
