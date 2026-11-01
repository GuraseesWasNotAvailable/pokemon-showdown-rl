// PSBot Lookahead: a window that plays rated gen9randombattle games on Pokemon Showdown with the
// newest network and the look-ahead search ("mix", the veto rule), on this laptop.
//
// It runs ladder.py (next to this exe) with the venv's Python as a child process without a
// console window, reads its output line by line and shows it. The network runs on the GPU; the
// search helpers run in WSL (foulplay_bridge.py starts them). ladder.py's "@app {json}" lines
// (--app-events) say when a game starts, every order sent, every result and rating change.
//
// Who decides (the list in the window): the bot itself (Lookahead, or the network alone), or a
// model of OpenAI that picks among the legal moves with the bot's numbers in front of it: Luna
// (gpt-6-luna, about 0.15 s and a hundredth of a cent a move) or Sol (gpt-6-sol, larger and
// slower: 1 to 5 s and about half a cent a move). "Read ahead" adds, after the search, its three
// favourite moves each played five turns forward and compared (ladder.py's arms mix-lines,
// luna-lines, sol-lines). Whenever a model gives no usable answer in time the bot's own move is played.
//
// The password is never stored, logged, put on a command line or in an environment block: it is
// written as one line to the child's standard input, which is then closed (ladder.py
// --password-stdin), and the field is cleared when the run ends. The OpenAI key of Luna and Sol is
// handled the same way: the second line of that standard input (ladder.py --key-stdin), never
// stored, cleared when the run ends; left empty, the bot uses the variable OPENAI_API_KEY when
// this computer has one. The username, the number of games, games at once and the mode are
// remembered in "PSBot Lookahead.settings" next to the exe.
//
// One bot per account. Only one window opens (a named mutex). Start looks for another ladder.py
// already playing on the same server (the Desktop's Look Play.cmd, a command line) and does not
// start next to it; ladder.py itself also takes a lock per server and account. During a run the
// window keeps looking and warns when another bot appears.
//
// The child is in a job object that kills it, and whatever it started, when this window goes,
// however it goes. The WSL helpers leave by themselves within 20 s when their wsl.exe is gone
// (bench/fp_bridge_server.py, --parent-watch beat); on a normal stop ladder.py stops them at once.
//
// Test options (not for play): --test-server PORT points the bot at ws://127.0.0.1:PORT instead
// of the public server, --test-user NAME fills the username, --test-foulplay-dir DIR gives the
// child FOULPLAY_DIR=DIR (a check of a missing Foul Play). The title says TEST then.
// --test-think [FILE] opens no window: it prints what every stop of the thinking slider passes to
// ladder.py in both search styles and both search modes (Thinking.Table), or writes it to FILE.
//
// Build: build-app.cmd (the C# compiler of .NET Framework 4, which ships with Windows).

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Globalization;
using System.IO;
using System.Management;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

static class Program
{
	public const string AppName = "PSBot Lookahead";

	[DllImport("user32.dll")]
	static extern bool SetProcessDPIAware();
	[DllImport("kernel32.dll")]
	static extern bool AttachConsole(int processId);

	// --test-think [FILE]: Thinking.Table at the default 2 games at once (3 helpers), into FILE or
	// onto the standard output (the console it was started from, when that is not redirected).
	static int PrintThinkTable(string file)
	{
		string text = Thinking.Table(3);
		try
		{
			if (!string.IsNullOrEmpty(file)) { File.WriteAllText(file, text, new UTF8Encoding(false)); return 0; }
			if (!Console.IsOutputRedirected) AttachConsole(-1);   // a window program has no console of its own
			Console.Out.Write(text);
			Console.Out.Flush();
			return 0;
		}
		catch (Exception) { return 1; }
	}

	[STAThread]
	static int Main(string[] argv)
	{
		try { SetProcessDPIAware(); } catch (Exception) { }
		Application.EnableVisualStyles();
		Application.SetCompatibleTextRenderingDefault(false);
		Options opts = new Options();
		for (int i = 0; i < argv.Length; i++)
		{
			string a = argv[i];
			string next = i + 1 < argv.Length ? argv[i + 1] : null;
			if (a == "--test-server" && next != null) { int p; if (int.TryParse(next, out p) && p > 0 && p < 65536) opts.TestPort = p; i++; }
			else if (a == "--test-user" && next != null) { opts.TestUser = next; i++; }
			else if (a == "--test-foulplay-dir" && next != null) { opts.TestFoulPlayDir = next; i++; }
			else if (a == "--test-memory-warn-gb" && next != null) { double g; if (double.TryParse(next, NumberStyles.Float, CultureInfo.InvariantCulture, out g)) opts.MemoryWarnGB = g; i++; }
			else if (a == "--test-replay" && next != null) { opts.ReplayFile = next; i++; }
			else if (a == "--test-snapshot" && next != null) { opts.SnapshotFile = next; i++; }
			else if (a == "--test-think")
			{
				opts.ThinkTable = true;
				if (next != null && !next.StartsWith("--")) { opts.ThinkFile = next; i++; }
			}
		}
		// The thinking table: no window, no mutex, nothing started.
		if (opts.ThinkTable) return PrintThinkTable(opts.ThinkFile);
		// A replay plays nothing and logs in nowhere: it may open beside the real window.
		if (opts.ReplayFile != null)
		{
			Application.Run(new MainForm(opts));
			return 0;
		}
		// One window at a time: two would mean two bots on one account.
		bool first;
		using (Mutex one = new Mutex(false, @"Local\PSBotLookahead", out first))
		{
			if (!first)
			{
				MessageBox.Show(AppName + " is already open. Use that window (it may be behind another window or minimised).",
					AppName, MessageBoxButtons.OK, MessageBoxIcon.Information);
				return 2;
			}
			try
			{
				Application.Run(new MainForm(opts));
				return 0;
			}
			catch (Exception e)
			{
				MessageBox.Show(e.ToString(), AppName, MessageBoxButtons.OK, MessageBoxIcon.Error);
				return 1;
			}
		}
	}
}

// A ladder.py on this computer that plays (not a graph or a history).
class OtherBot
{
	public int Pid;
	public string Server;   // "public" or "local:PORT"
	public string User;     // null: not on its command line (PS_USERNAME, or typed at its prompt)
	public string Started;

	static string ArgValue(string cmd, string name)
	{
		Match m = Regex.Match(cmd, @"(?:^|\s)" + Regex.Escape(name) + @"(?:=|\s+)(?:""([^""]*)""|(\S+))");
		if (!m.Success) return null;
		return m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value;
	}

	public static string ToId(string name)
	{
		return name == null ? null : Regex.Replace(name.ToLowerInvariant(), "[^a-z0-9]", "");
	}

	// Every playing ladder.py except ownPid and its children (the venv's python.exe is a launcher
	// that starts the real interpreter with the same command line: the pair is reported once).
	public static List<OtherBot> Find(int ownPid)
	{
		List<int[]> rows = new List<int[]>();
		Dictionary<int, string> cmdOf = new Dictionary<int, string>();
		Dictionary<int, string> startOf = new Dictionary<int, string>();
		using (ManagementObjectSearcher s = new ManagementObjectSearcher(
			"SELECT ProcessId, ParentProcessId, CommandLine, CreationDate FROM Win32_Process WHERE Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'py.exe'"))
		using (ManagementObjectCollection all = s.Get())
		{
			foreach (ManagementBaseObject mo in all)
			{
				string cmd = mo["CommandLine"] as string;
				if (cmd == null || !Regex.IsMatch(cmd, @"(^|[\s""\\/])ladder\.py([\s""]|$)", RegexOptions.IgnoreCase)) continue;
				if (Regex.IsMatch(cmd, @"\s--(graph|history)(\s|=|$)")) continue;
				int pid = Convert.ToInt32(mo["ProcessId"], CultureInfo.InvariantCulture);
				int ppid = Convert.ToInt32(mo["ParentProcessId"], CultureInfo.InvariantCulture);
				if (ownPid > 0 && (pid == ownPid || ppid == ownPid)) continue;
				rows.Add(new int[] { pid, ppid });
				cmdOf[pid] = cmd;
				string started = "?";
				try { started = ManagementDateTimeConverter.ToDateTime(mo["CreationDate"] as string).ToString("HH:mm", CultureInfo.InvariantCulture); }
				catch (Exception) { }
				startOf[pid] = started;
			}
		}
		List<OtherBot> found = new List<OtherBot>();
		foreach (int[] r in rows)
		{
			if (cmdOf.ContainsKey(r[1])) continue;   // the interpreter of a launcher already listed
			OtherBot b = new OtherBot();
			b.Pid = r[0];
			b.Server = ArgValue(cmdOf[r[0]], "--server") ?? "public";
			b.User = ArgValue(cmdOf[r[0]], "--user");
			b.Started = startOf[r[0]];
			found.Add(b);
		}
		return found;
	}

	// Plays on this server as this account, or as an account that cannot be told.
	public bool Clashes(string server, string user)
	{
		return Server == server && (User == null || ToId(User) == ToId(user));
	}
}

class Options
{
	public int TestPort;              // 0: the public server
	public string TestUser;
	public string TestFoulPlayDir;
	public double MemoryWarnGB = 8.0;   // --test-memory-warn-gb: another bar, to see the warning in a test
	public string ReplayFile;           // --test-replay: "@app" lines of a run, shown as if a run printed them (no run is started)
	public string SnapshotFile;         // --test-snapshot: with --test-replay, a PNG of the window after them, then the window closes
	public bool ThinkTable;             // --test-think: print the thinking table (Thinking.Table) and end, no window
	public string ThinkFile;            // --test-think FILE: write it there instead
	public bool Test { get { return TestPort > 0; } }
}

// ------------------------------------------------------------------------------------------
// The thinking time and the search style, as ladder.py's search arguments: one pure function
// (Thinking.Plan), so --test-think prints its whole table without a window.
//
// Deeper is the search as it was: Foul Play guesses 2 opponent teams a move (4 at half the time
// early in a game) and searches each one longer as the time grows. Wider keeps each team's search
// at 500 ms (the 1 s setting's) and guesses that many times more teams, searched one after another
// (ladder.py --player mix-deep, --deep-worlds). On 474 ladder games 5 s searched deeper won no
// more than 1 s, with the same 2.2 teams a move (reports/2026-10-16/SEARCH_TIME.md); more teams is
// the untried idea.
// ------------------------------------------------------------------------------------------
sealed class ThinkPlan
{
	public double Seconds;          // the slider: about this long a move
	public bool Wider;              // the style picked (Wider below about 2 s is Deeper: Worlds 1)
	public bool Guarded;            // the guarded mode: the search picks among the moves the network gives 3% or more
	public string Player;           // --player
	public int SearchMs;            // --search-ms: Deeper's search per guessed team; with Wider today's 500 ms, which a game falls back to
	public double Deadline;         // --deadline: the cap on a decision of that search (the network's move after it)
	public int Worlds = 1;          // Wider: --deep-worlds, times as many guessed teams (1: no deep budget, the plan is Deeper's)
	public int DeepSearchMs;        // Wider: --deep-search-ms, the search per guessed team
	public double DeepDeadline;     // Wider: --deep-deadline
	public double Trust;            // Wider in the guarded mode: --trust (0: not passed; Deeper's guarded player is mix-trust)

	public bool Wide { get { return Worlds > 1; } }
	public int TeamMs { get { return Wide ? DeepSearchMs : SearchMs; } }
	public double Cap { get { return Wide ? DeepDeadline : Deadline; } }
	public int Teams { get { return 2 * Worlds; } }     // guessed teams a move; twice as many at half the time early in a game
	public double AboutSeconds { get { return Thinking.DecisionPerSearch * TeamMs * Worlds / 1000.0; } }
	public double LeastDeadline { get { return Thinking.MinimumDeadline(TeamMs, Worlds); } }

	// ladder.py's search arguments. Deeper's are the window's from before the slider, in their order.
	public List<string> Args(int helpers)
	{
		List<string> a = new List<string> { "--player", Player, "--mix-rule", "veto", "--search-ms", SearchMs.ToString(CultureInfo.InvariantCulture),
			"--helpers", helpers.ToString(CultureInfo.InvariantCulture), "--deadline", Thinking.Tenths(Deadline) };
		if (!Wide) return a;
		// One worker and one thread a world: the same cores as today, the teams one after another in
		// the battle's own helper (workers would fork more processes per game on the laptop).
		a.AddRange(new string[] { "--deep-search-ms", DeepSearchMs.ToString(CultureInfo.InvariantCulture),
			"--deep-worlds", Worlds.ToString(CultureInfo.InvariantCulture), "--deep-threads", "1", "--deep-workers", "1",
			"--deep-deadline", Thinking.Tenths(DeepDeadline) });
		if (Trust > 0) a.AddRange(new string[] { "--trust", Trust.ToString("0.0#", CultureInfo.InvariantCulture) });
		return a;
	}

	// The search line of the window, in plain words: short, so that the guarded mode's note and the
	// last game's line fit under it in the label's 4 lines (twice the teams at half the time early in
	// a game is in the style buttons' tooltips).
	public string Describe()
	{
		string s = Wider && !Wide ? "Deeper (Wider starts at " + Thinking.Seconds(Thinking.WiderFrom()) + " s): " : (Wide ? "Wider: " : "Deeper: ");
		return s + Teams + " guessed opponent teams, " + TeamMs + " ms each. About " + AboutSeconds.ToString("0.0", CultureInfo.InvariantCulture) +
			" s a move, at most " + Thinking.Tenths(Cap) + " s, else the network's move.";
	}

	// The start line of the log.
	public string LogText()
	{
		return Wide ? "search " + DeepSearchMs + " ms on " + Worlds + "x the guessed teams" : "search " + SearchMs + " ms";
	}
}

static class Thinking
{
	// The slider, in seconds a move. Its top is the longest move whose cap leaves OrderRoom under the
	// 10 s of the battle timer (Deeper 3100 ms, cap 8.2 s; Wider 6x, cap 8.0 s): stops above it would
	// play the same as 7 s.
	public const double Least = 0.5, Most = 7.0, Step = 0.5, Default = 1.0;
	// A decision takes about 2.26 x the search per guessed team (10 ladder games of the arm C network,
	// 2026-10-15: 500 ms -> 1.13 s; 474 games to 2026-10-16: 1.17 s at 500 ms, 4.98 s at 2200 ms).
	public const double DecisionPerSearch = 2.26;
	public const int TodayMs = 500;             // the 1 s setting; Wider's search per team, and what a Wider game falls back to
	public const double TodayDeadline = 2.5;    // its cap, and the least cap of any setting
	public const double GuardTrust = 0.03;      // search_player.ARM_FIXES["mix-trust"]
	public const double TimerCap = 10.0;        // an order in 10 s or more after the request costs two ticks of the battle timer
	// What comes after the cap before the server has the order: ladder.py's order spacing may hold it
	// up to --order-max-wait-s (1.5 s, OrderGuard), and the request and the order travel between the
	// laptop and the server (about 0.3 s both ways). search_player's DEEP_BUDGET keeps its 7.5 s
	// deadline for the same reason.
	public const double OrderRoom = 1.8;

	// Whether a cap leaves OrderRoom under the battle timer's 10 s.
	public static bool Fits(double cap) { return cap + OrderRoom <= TimerCap + 1e-9; }

	public static string Tenths(double v) { return v.ToString("0.0", CultureInfo.InvariantCulture); }
	public static string Seconds(double v) { return v.ToString("0.#", CultureInfo.InvariantCulture); }

	// search_player.minimum_deadline with one worker: what the helper reckons for the search (2.5 x
	// ms x worlds + 20 ms), the 50 ms reserve and 0.3 s for the wait in front of it. A search that
	// cannot end by the deadline is never started.
	public static double MinimumDeadline(int ms, int worlds) { return 2.5 * ms * worlds / 1000.0 + 0.02 + 0.05 + 0.3; }

	// The cap: that least deadline and 50 ms more, up to the next tenth of a second, never under
	// today's 2.5 s. It gives the 2.5, 2.7 and 6.0 s of 500, 900 and 2200 ms.
	public static double CapFor(int ms, int worlds)
	{
		return Math.Max(TodayDeadline, Math.Ceiling((MinimumDeadline(ms, worlds) + 0.05) * 10.0 - 1e-6) / 10.0);
	}

	// A slider value: to the half second, within 0.5 to 7 s.
	public static double Snap(double seconds)
	{
		if (double.IsNaN(seconds) || double.IsInfinity(seconds)) return Default;
		double v = Math.Round(seconds / Step, MidpointRounding.AwayFromZero) * Step;
		return Math.Max(Least, Math.Min(Most, v));
	}

	// The saved "think": seconds ("1.5"), or the list's keys of before ("1s", "2s", "5s": 1, 2 and 5 s).
	public static double ParseSeconds(string saved)
	{
		string s = (saved ?? "").Trim();
		if (s.EndsWith("s")) s = s.Substring(0, s.Length - 1);
		double v;
		if (!double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v)) return Default;
		return Snap(v);
	}

	// Deeper's search per guessed team for about `seconds` a move: the time over 2.26, to 100 ms,
	// with a cap that Fits. 1 s keeps today's 500 ms (1.13 s a move); 2 s and 5 s give 900 and 2200 ms.
	public static int DeeperMs(double seconds)
	{
		if (Math.Abs(seconds - 1.0) < 1e-9) return TodayMs;
		int ms = Math.Max(100, 100 * (int)Math.Round(seconds * 10.0 / DecisionPerSearch, MidpointRounding.AwayFromZero));
		while (ms > 100 && !Fits(CapFor(ms, 1))) ms -= 100;
		return ms;
	}

	// Wider's times as many guessed teams: the count whose move (2.26 x 500 ms a team) comes nearest
	// `seconds`, at least 1, and fewer while the cap would not fit. Its cap (at most 8.0 s) also stays
	// under ladder.py's --deep-max-decision-s (9.0 s), past which a game goes back to the 1 s search.
	public static int WiderWorlds(double seconds)
	{
		int w = Math.Max(1, (int)Math.Round(seconds * 1000.0 / (DecisionPerSearch * TodayMs), MidpointRounding.AwayFromZero));
		while (w > 1 && !Fits(CapFor(TodayMs, w))) w--;
		return w;
	}

	// The least slider value at which Wider guesses more teams than Deeper.
	public static double WiderFrom()
	{
		for (double t = Least; t <= Most + 1e-9; t += Step)
			if (WiderWorlds(t) > 1) return t;
		return Most;
	}

	// The plan for about `seconds` a move in a style, for the mode's ladder.py player ("mix", or
	// "mix-trust" for the guarded mode: MainForm.PlayerOf).
	public static ThinkPlan Plan(double seconds, bool wider, string player)
	{
		ThinkPlan p = new ThinkPlan();
		p.Seconds = Snap(seconds);
		p.Wider = wider;
		p.Guarded = player == "mix-trust";
		int w = wider ? WiderWorlds(p.Seconds) : 1;
		if (w <= 1)
		{
			p.Player = player;
			p.SearchMs = DeeperMs(p.Seconds);
			p.Deadline = CapFor(p.SearchMs, 1);
			return p;
		}
		p.Player = "mix-deep";              // mix's settings with a budget of its own (search_player.DEEP_ARM)
		p.SearchMs = TodayMs;
		p.Deadline = TodayDeadline;
		p.Worlds = w;
		p.DeepSearchMs = TodayMs;
		p.DeepDeadline = CapFor(TodayMs, w);
		if (p.Guarded) p.Trust = GuardTrust;  // mix-deep is no ARM_FIXES arm: the guard comes with ladder.py --trust
		return p;
	}

	// --test-think: every slider stop in both styles and both search modes, one line each; the
	// arguments are what the window passes with `helpers` (games at once + 1, at most 4).
	public static string Table(int helpers)
	{
		StringBuilder sb = new StringBuilder();
		sb.Append("# PSBot Lookahead --test-think: the thinking slider and the search style as ladder.py's search arguments\r\n");
		sb.Append("# helpers " + helpers + " (games at once + 1, at most 4); a move takes about " + DecisionPerSearch.ToString(CultureInfo.InvariantCulture) +
			" x the search per guessed team\r\n");
		sb.Append("# teams: guessed opponent teams a move (twice as many at half the time early in a game); ms: the search per team\r\n");
		sb.Append("# about: the expected move; cap: the deadline passed; floor: search_player.minimum_deadline of that search\r\n");
		sb.Append("# " + Pad("seconds", 8) + Pad("style", 7) + Pad("mode", 10) + Pad("player", 10) + Pad("teams", 6) + Pad("ms", 6) +
			Pad("about", 6) + Pad("cap", 6) + Pad("floor", 6) + "args\r\n");
		for (int i = (int)Math.Round(Least / Step); i <= (int)Math.Round(Most / Step); i++)
		{
			double t = i * Step;
			foreach (bool wide in new bool[] { false, true })
				foreach (string mode in new string[] { "lookahead", "guarded" })
				{
					ThinkPlan p = Plan(t, wide, mode == "guarded" ? "mix-trust" : "mix");
					sb.Append("  " + Pad(Seconds(t), 8) + Pad(wide ? "wider" : "deeper", 7) + Pad(mode, 10) + Pad(p.Player, 10) +
						Pad(p.Teams.ToString(CultureInfo.InvariantCulture), 6) + Pad(p.TeamMs.ToString(CultureInfo.InvariantCulture), 6) +
						Pad(p.AboutSeconds.ToString("0.00", CultureInfo.InvariantCulture), 6) + Pad(Tenths(p.Cap), 6) +
						Pad(p.LeastDeadline.ToString("0.00", CultureInfo.InvariantCulture), 6) + string.Join(" ", p.Args(helpers).ToArray()) + "\r\n");
				}
		}
		return sb.ToString();
	}

	static string Pad(string s, int width) { return s.PadRight(width - 1) + " "; }
}

// ------------------------------------------------------------------------------------------
// A job object: every process in it is killed when the job's last handle is closed, which
// happens when this app ends, however it ends.
// ------------------------------------------------------------------------------------------
sealed class Job : IDisposable
{
	[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
	static extern IntPtr CreateJobObject(IntPtr attributes, string name);
	[DllImport("kernel32.dll")]
	static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);
	[DllImport("kernel32.dll")]
	static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
	[DllImport("kernel32.dll")]
	static extern bool TerminateJobObject(IntPtr job, uint exitCode);
	[DllImport("kernel32.dll")]
	static extern bool CloseHandle(IntPtr handle);

	[StructLayout(LayoutKind.Sequential)]
	struct BasicLimits
	{
		public long PerProcessUserTimeLimit;
		public long PerJobUserTimeLimit;
		public uint LimitFlags;
		public UIntPtr MinimumWorkingSetSize;
		public UIntPtr MaximumWorkingSetSize;
		public uint ActiveProcessLimit;
		public UIntPtr Affinity;
		public uint PriorityClass;
		public uint SchedulingClass;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct IoCounters
	{
		public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
		public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct ExtendedLimits
	{
		public BasicLimits Basic;
		public IoCounters Io;
		public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
	}

	const int ExtendedLimitInformation = 9;
	const uint KillOnJobClose = 0x2000;
	IntPtr handle;

	public Job()
	{
		handle = CreateJobObject(IntPtr.Zero, null);
		if (handle == IntPtr.Zero) throw new Win32Exception();
		ExtendedLimits info = new ExtendedLimits();
		info.Basic.LimitFlags = KillOnJobClose;
		int size = Marshal.SizeOf(typeof(ExtendedLimits));
		IntPtr ptr = Marshal.AllocHGlobal(size);
		try
		{
			Marshal.StructureToPtr(info, ptr, false);
			if (!SetInformationJobObject(handle, ExtendedLimitInformation, ptr, (uint)size)) throw new Win32Exception();
		}
		finally { Marshal.FreeHGlobal(ptr); }
	}

	public void Add(Process p)
	{
		if (!AssignProcessToJobObject(handle, p.Handle)) throw new Win32Exception();
	}

	public void Kill()
	{
		if (handle != IntPtr.Zero) TerminateJobObject(handle, 1);
	}

	public void Dispose()
	{
		if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; }   // kills what is left in it
	}
}

static class Memory
{
	[StructLayout(LayoutKind.Sequential)]
	struct MemoryStatusEx
	{
		public uint Length, MemoryLoad;
		public ulong TotalPhys, AvailPhys, TotalPageFile, AvailPageFile, TotalVirtual, AvailVirtual, AvailExtendedVirtual;
	}

	[DllImport("kernel32.dll")]
	static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx status);

	// Free physical memory in GB, -1 when it cannot be read.
	public static double FreeGB()
	{
		MemoryStatusEx s = new MemoryStatusEx();
		s.Length = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
		if (!GlobalMemoryStatusEx(ref s)) return -1;
		return s.AvailPhys / (1024.0 * 1024.0 * 1024.0);
	}
}

class Checkpoint
{
	public string Path;
	public int Update;
	public DateTime Written;
	public bool Fresh;   // written in the last two minutes: it may still be being copied
	public bool Complete;   // ends with a zip archive's end record (an interrupted copy does not)
	public bool Baseline;   // baseline_update_N.pt: a network of the search branch's baseline run, its own numbering
	public bool ArmC;       // armC-full_update_N_xM.pt: the full run from arm C, in its own folder, its own numbering
	public int Millions;    // arm C and v2: the decisions it had trained on, in millions (from the file name)
	public string V2Arm;    // v2-<arm>_update_N_xM.pt: an arm of the v2 training waves (null otherwise)
	public int Line { get { return V2Arm != null ? 3 : ArmC ? 2 : Baseline ? 1 : 0; } }   // the list's order: main line, baseline run, arm C, v2
	public string Label;

	// A network file is a zip archive (torch.save): a whole one ends with the end-of-central-
	// directory record (PK 5 6) within its last 64 KB. Reads only that tail.
	public static bool LooksComplete(string path)
	{
		try
		{
			using (FileStream f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
			{
				long n = f.Length;
				if (n < 1024) return false;
				int len = (int)Math.Min(n, 65536 + 22);
				byte[] tail = new byte[len];
				f.Seek(n - len, SeekOrigin.Begin);
				int read = 0;
				while (read < len)
				{
					int r = f.Read(tail, read, len - read);
					if (r <= 0) return false;
					read += r;
				}
				for (int i = len - 22; i >= 0; i--)
					if (tail[i] == 0x50 && tail[i + 1] == 0x4b && tail[i + 2] == 5 && tail[i + 3] == 6) return true;
				return false;
			}
		}
		catch (Exception) { return false; }
	}
	public override string ToString() { return Label; }
}

// One of the small dark panels on the right of the window (Elo, attention, neurons): it draws
// itself with the form's painter, double-buffered so a move a second does not flicker.
class ViewPanel : Panel
{
	public static readonly Color Back = Color.FromArgb(16, 20, 26);
	public static readonly Color Ink = Color.FromArgb(214, 220, 228);
	public static readonly Color Dim = Color.FromArgb(128, 138, 150);
	public static readonly Color Rule = Color.FromArgb(52, 60, 72);
	public Action<Graphics, Rectangle> Painter;

	public ViewPanel()
	{
		SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
		BackColor = Back;
	}

	protected override void OnPaint(PaintEventArgs e)
	{
		base.OnPaint(e);
		e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
		e.Graphics.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
		try { if (Painter != null) Painter(e.Graphics, ClientRectangle); }
		catch (Exception ex)   // a drawing error must never take the window down
		{
			using (Brush b = new SolidBrush(Color.OrangeRed)) e.Graphics.DrawString("could not draw: " + ex.Message, Font, b, 4, 4);
		}
	}
}

// The network's latest move as the window shows it: ladder.py's "net" line (net_viz.py).
class NetFrame
{
	public string Tag;
	public int Game, Turn;
	public string[] Own = new string[0], Foe = new string[0];
	public int OwnActive = -1, FoeActive = -1;
	public int Layers;
	public double[][] Att;      // a row per layer and one for the readout: the share on each of 14 columns
	public byte[] Neu;          // the same rows: d_ff neurons each, 0..255 of the row's largest value
	public double[] NeuMax;     // that largest value, per row
	public bool Ended;          // its game is over: the last move stays on show until the next game's
	// The diagram (net_viz.diagram): each row's strongest neurons, the direct weights between them,
	// the network's move probabilities and its likeliest moves with the weights into them.
	public int[][] Top;
	public double[][][] Edges;  // Edges[r][j][i]: from neuron i of row r to neuron j of row r + 1, -1..1
	public double[] Probs;
	public int[] Acts;
	public string[] ActNames;
	public double[][] OutEdges; // OutEdges[a][i]: from neuron i of the last row to move a
	public double[] Params;     // parameters per block: input encoder, each layer, readout, policy head (net_viz.param_blocks)
}

// The three choices when the window is closed during a run.
class CloseDialog : Form
{
	public const int Finish = 1, StopNow = 2, Cancel = 0;
	public int Choice = Cancel;

	public CloseDialog(int playing)
	{
		// The designer's order: controls first at 96 DPI, then the scaling mode, then one layout
		// pass that scales them all to this screen.
		SuspendLayout();
		Text = Program.AppName;
		FormBorderStyle = FormBorderStyle.FixedDialog;
		MaximizeBox = MinimizeBox = false;
		StartPosition = FormStartPosition.CenterParent;
		Font = new Font("Segoe UI", 9F);
		ClientSize = new Size(430, 196);
		Label l = new Label();
		l.Text = playing > 0
			? "The bot is playing " + playing + (playing == 1 ? " game" : " games") + ". What should happen to " + (playing == 1 ? "it" : "them") + "?"
			: "The bot is running. Stop it?";
		l.SetBounds(16, 14, 400, 40);
		Controls.Add(l);
		Button finish = MakeButton("Finish the games in progress, then close (recommended)", 60, Finish);
		Button now = MakeButton("Stop now and close (games in progress are lost on time)", 98, StopNow);
		Button cancel = MakeButton("Cancel: keep playing", 136, Cancel);
		AcceptButton = finish;
		CancelButton = cancel;
		AutoScaleDimensions = new SizeF(96F, 96F);
		AutoScaleMode = AutoScaleMode.Dpi;
		ResumeLayout(false);
		PerformLayout();
	}

	Button MakeButton(string text, int y, int choice)
	{
		Button b = new Button();
		b.Text = text;
		b.SetBounds(16, y, 398, 30);
		b.Click += delegate { Choice = choice; Close(); };
		Controls.Add(b);
		return b;
	}
}

class MainForm : Form
{
	// Where things are, found next to the app (MainForm sets them from appDir): Python is the .venv in
	// this folder or the one above it, else the first python.exe on PATH; every network file goes in
	// weights\ next to the exe. The settings file overrides each (python=, checkpoints=, armc=, v2=).
	string pythonExe = "";
	string checkpointsDir = "";
	string armcDir = "";   // the full run from arm C, kept apart
	string v2Dir = "";     // the v2 training runs' exported networks (v2-<arm>_update_N_xM.pt)

	// What each v2 arm trained with, in plain words for the list.
	static readonly Dictionary<string, string> V2Names = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
	{
		{ "a0", "control, no new technique" },
		{ "readhead", "switch-reading reward + opponent prediction" },
		{ "read", "switch-reading reward" },
		{ "head", "opponent prediction" },
		{ "setupw2", "set-up practice, boosted positions weighted" },
		{ "setup15", "set-up practice" },
		{ "fp10", "games against Foul Play" },
		{ "musethead", "abilities and items" },
		{ "scratch", "from scratch, every technique" },
		{ "topstart", "top players first, every technique" },
		{ "league", "from 24381, against frozen past networks" },
	};
	// Thinking time per move with the search (Lookahead, guarded): the slider, 0.5 to 7 s in half
	// seconds, and the search style; Thinking.Plan turns them into ladder.py's search arguments.
	double ThinkSeconds { get { return thinkBar == null ? Thinking.Default : thinkBar.Value * Thinking.Step; } }
	bool WiderPicked { get { return widerButton != null && widerButton.Checked; } }
	string lastThinking = "";   // the last game's measured decision time and search iterations (NoteThinking)
	string lastThinkingLine = "";   // the same in one line of the search label
	const int MaxHelpers = 4;
	double LowMemoryGB = 8.0;

	readonly Options opts;
	readonly string appDir;
	readonly string settingsFile;
	readonly string runsDir;
	readonly JavaScriptSerializer json = new JavaScriptSerializer();
	readonly Dictionary<string, string> settings = new Dictionary<string, string>();

	TextBox userBox, passBox, keyBox, logBox;
	NumericUpDown gamesBox, parallelBox;
	ComboBox modeBox, netBox, effortBox;
	TrackBar thinkBar;
	Label thinkLabel;
	Panel styleBox;
	RadioButton deeperButton, widerButton;
	CheckBox allNetsBox;
	Label keyLabel, keyTitle;
	Label foulPlayLabel, searchLabel, ramLabel;
	Label stateLabel, playedLabel, ratingLabel, progressLabel, lastLabel;
	Button startButton, stopAfterButton, stopNowButton, graphButton, fpCheckButton;
	System.Windows.Forms.Timer tick;
	bool netPicked;   // a network was picked by hand: Start keeps it (else Start takes the newest whole one)

	// The run.
	Process child;
	Job childJob;
	string stopFile, stopNowFile;
	StreamWriter runLog;
	readonly ManualResetEvent outDone = new ManualResetEvent(true), errDone = new ManualResetEvent(true);
	bool running, sawStart, sawFinished, stoppingAfter, stoppingNow, closeWhenDone;
	DateTime stopNowAt = DateTime.MinValue;
	int gamesTarget, wins, losses, abandoned;
	int? ratingStart, ratingNow;
	string phase = "";
	string lastError;
	readonly List<string> lastStderr = new List<string>();
	readonly Dictionary<string, GameInfo> games = new Dictionary<string, GameInfo>();   // in progress, by battle tag
	readonly Dictionary<string, int> gameNumbers = new Dictionary<string, int>();

	// Foul Play in WSL: null while not checked or being checked (fpChecking).
	bool? foulPlayOk;
	bool fpChecking;
	string foulPlayWhy = "";
	bool graphRunning;

	// The column on the right (BuildWindow): this run's Elo and the network at work.
	const int ViewColumn = 312;
	ViewPanel eloView, neuView;
	SplitContainer viewSplit;                           // Elo above, neurons below: the bar between them moves
	Panel viewGrip;                                     // the column's left edge: drag it to make the column wider
	GroupBox statusBox;
	readonly List<int> eloTrail = new List<int>();     // this run: the rating before its first game, then after each game
	readonly List<bool> eloWon = new List<bool>();     // eloWon[i]: the game that led to eloTrail[i + 1]
	NetFrame netNow;                                    // the move on show
	string followTag;                                   // the game it follows: one at a time, so games at once do not flicker
	double[] neuScale;                                  // per row: a slowly falling top of the neurons' values, so brightness compares across moves
	bool runUsesNet = true;
	readonly Font viewTitle = new Font("Segoe UI", 9F, FontStyle.Bold);
	readonly Font viewSmall = new Font("Segoe UI", 7.5F);
	// The diagram's sweep after each move (3Blue1Brown style): the rows light up left to right.
	const double SweepMs = 950;
	DateTime netAt = DateTime.MinValue;
	bool animate = true;                                // off for --test-replay, so a snapshot shows the whole picture
	System.Windows.Forms.Timer netTick;
	bool view3d = true;                                 // the neurons panel as 3D blocks (the "3D" pill on it; settings netview3d)
	RectangleF togglePill;                              // where the pill was drawn, for the click

	// Another bot on this computer: what this run plays as, and the ones already warned of.
	string runServer, runUser;
	bool scanning;
	int scanTicks;
	readonly HashSet<int> warnedBots = new HashSet<int>();

	class GameInfo { public int N; public string Opponent; public int Turn; }

	public MainForm(Options o)
	{
		opts = o;
		LowMemoryGB = o.MemoryWarnGB;
		appDir = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
		settingsFile = System.IO.Path.Combine(appDir, Program.AppName + (opts.Test ? ".test" : "") + ".settings");
		runsDir = System.IO.Path.Combine(appDir, "runs", "app");
		pythonExe = DefaultPython(appDir);
		checkpointsDir = System.IO.Path.Combine(appDir, "weights");
		armcDir = System.IO.Path.Combine(appDir, "weights", "armc");
		v2Dir = System.IO.Path.Combine(appDir, "weights");
		LoadSettings();
		BuildWindow();
		FillCheckpoints(false);
		if (ModeKey == "guarded") SelectBestNetwork();
		UpdateRam();
		UpdateButtons();
		tick = new System.Windows.Forms.Timer();
		tick.Interval = 3000;
		tick.Tick += delegate { UpdateRam(); CheckStopNowTimeout(); ScanDuringRun(); };
		tick.Start();
		Shown += delegate { if (opts.ReplayFile != null) Replay(); else FirstFoulPlayCheck(); };
		FormClosing += OnClosing;
	}

	string ServerArg() { return opts.Test ? "local:" + opts.TestPort : "public"; }

	// The Python that runs ladder.py: a .venv in the app's folder or the folder above it, else the first
	// python.exe on PATH (not the Microsoft Store stub). Returns the first venv path when none exists, so
	// Start says where it looked.
	static string DefaultPython(string dir)
	{
		string here = System.IO.Path.Combine(dir, ".venv", "Scripts", "python.exe");
		string parent = System.IO.Path.GetDirectoryName(dir);
		string[] venvs = parent == null ? new string[] { here } : new string[] { here, System.IO.Path.Combine(parent, ".venv", "Scripts", "python.exe") };
		foreach (string v in venvs) if (File.Exists(v)) return v;
		foreach (string p in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';'))
		{
			try
			{
				string d = p.Trim().Trim('"');
				if (d.Length == 0) continue;
				string c = System.IO.Path.Combine(d, "python.exe");
				if (File.Exists(c) && c.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) < 0) return c;
			}
			catch (ArgumentException) { }
		}
		return here;
	}
	static string ServerText(string server) { return server == "public" ? "play.pokemonshowdown.com" : "the test server " + server; }

	// --------------------------------------------------------------------------------------
	// The window
	// --------------------------------------------------------------------------------------

	void BuildWindow()
	{
		// The designer's order: every control at 96 DPI with layout suspended, then the scaling
		// mode, then one layout pass that scales them all to this screen (150% on the laptop).
		SuspendLayout();
		Text = Program.AppName + (opts.Test ? "  -  TEST server 127.0.0.1:" + opts.TestPort : "");
		Font = new Font("Segoe UI", 9F);
		ClientSize = new Size(940 + ViewColumn, 796);
		StartPosition = FormStartPosition.CenterScreen;

		GroupBox run = new GroupBox();
		run.Text = "Run";
		run.SetBounds(12, 8, 430, 434);
		Controls.Add(run);
		int lx = 14, cx = 156, cw = 256;
		AddLabel(run, "Showdown username", lx, 28);
		userBox = new TextBox();
		userBox.Name = "user";
		userBox.SetBounds(cx, 25, cw, 24);
		userBox.Text = opts.Test && !string.IsNullOrEmpty(opts.TestUser) ? opts.TestUser : Get("user", "");
		run.Controls.Add(userBox);
		AddLabel(run, "Password", lx, 58);
		passBox = new TextBox();
		passBox.Name = "password";
		passBox.SetBounds(cx, 55, cw, 24);
		passBox.UseSystemPasswordChar = true;
		run.Controls.Add(passBox);
		AddLabel(run, "Games to play", lx, 90);
		gamesBox = new NumericUpDown();
		gamesBox.Name = "games";
		gamesBox.SetBounds(cx, 87, 80, 24);
		gamesBox.Minimum = 1; gamesBox.Maximum = 1000;
		gamesBox.Value = Clamp(GetInt("games", 20), 1, 1000);
		run.Controls.Add(gamesBox);
		AddLabel(run, "Games at once", lx, 122);
		parallelBox = new NumericUpDown();
		parallelBox.Name = "parallel";
		parallelBox.SetBounds(cx, 119, 80, 24);
		parallelBox.Minimum = 1; parallelBox.Maximum = 3;
		parallelBox.Value = Clamp(GetInt("parallel", 2), 1, 3);
		parallelBox.ValueChanged += delegate { UpdateSearchLabel(); };
		run.Controls.Add(parallelBox);
		AddLabel(run, "Who plays", lx, 154);
		modeBox = new ComboBox();
		modeBox.Name = "mode";
		modeBox.DropDownStyle = ComboBoxStyle.DropDownList;
		modeBox.SetBounds(cx, 151, cw, 24);
		foreach (string text in ModeTexts) modeBox.Items.Add(text);
		modeBox.SelectedIndex = Math.Max(0, Array.IndexOf(ModeKeys, SavedMode(Get("mode", "lookahead"))));
		run.Controls.Add(modeBox);
		// How much a Claude model thinks before each move (ladder.py --claude-effort). Only for those.
		effortBox = new ComboBox();
		effortBox.Name = "effort";
		effortBox.DropDownStyle = ComboBoxStyle.DropDownList;
		effortBox.SetBounds(cx, 180, cw, 24);
		foreach (string text in EffortTexts) effortBox.Items.Add(text);
		effortBox.SelectedIndex = Math.Max(0, Array.IndexOf(EffortKeys, Get("effort", "low")));
		effortBox.SelectedIndexChanged += delegate { UpdateSearchLabel(); };
		run.Controls.Add(effortBox);
		// How long the bot thinks per move with the search, and how it spends the time (Lookahead and
		// guarded only). In the place of the list above and of the API key's row, which no search mode
		// uses: one or the other is shown (UpdateButtons).
		thinkLabel = new Label();
		thinkLabel.Name = "thinktext";
		thinkLabel.SetBounds(lx, 176, 140, 34);
		run.Controls.Add(thinkLabel);
		thinkBar = new TrackBar();
		thinkBar.Name = "think";
		thinkBar.AutoSize = false;
		thinkBar.SetBounds(cx - 4, 177, cw + 8, 30);
		thinkBar.Minimum = (int)Math.Round(Thinking.Least / Thinking.Step);   // half seconds
		thinkBar.Maximum = (int)Math.Round(Thinking.Most / Thinking.Step);
		thinkBar.SmallChange = 1;
		thinkBar.LargeChange = 2;
		// A tick on every stop: the ticks count from the minimum (0.5 s), so one every second stop
		// would sit on the half seconds and miss 1, 2 and 5 s.
		thinkBar.TickFrequency = 1;
		thinkBar.TickStyle = TickStyle.BottomRight;
		thinkBar.Value = (int)Math.Round(Thinking.ParseSeconds(Get("think", "1")) / Thinking.Step);
		thinkBar.ValueChanged += delegate { UpdateSearchLabel(); };
		new ToolTip().SetToolTip(thinkBar, "About how long the bot thinks a move: " + Thinking.Seconds(Thinking.Least) + " to " + Thinking.Seconds(Thinking.Most) + " s.");
		run.Controls.Add(thinkBar);
		styleBox = new Panel();
		styleBox.Name = "searchstyle";
		styleBox.SetBounds(lx, 210, 404, 46);
		deeperButton = new RadioButton();
		deeperButton.Name = "deeper";
		deeperButton.Text = "Deeper: search each guessed opponent team longer";
		deeperButton.SetBounds(0, 2, 400, 20);
		widerButton = new RadioButton();
		widerButton.Name = "wider";
		widerButton.Text = "Wider: guess more opponent teams, same depth each";
		widerButton.SetBounds(0, 24, 400, 20);
		widerButton.Checked = Get("searchstyle", "deeper") == "wider";
		deeperButton.Checked = !widerButton.Checked;
		widerButton.CheckedChanged += delegate { UpdateSearchLabel(); };
		new ToolTip().SetToolTip(deeperButton, "The search as before: 2 guessed opponent teams a move (4 at half the time early in a game),\n" +
			"each searched longer as the time grows.");
		new ToolTip().SetToolTip(widerButton, "New, not yet tried on the ladder: more guessed opponent teams a move (twice as many at half the time\n" +
			"early in a game), each searched 500 ms as at 1 s, one after another. From 2 s a move.");
		styleBox.Controls.Add(deeperButton);
		styleBox.Controls.Add(widerButton);
		run.Controls.Add(styleBox);
		keyTitle = new Label();
		keyTitle.Name = "keytitle";
		keyTitle.Text = "API key";
		keyTitle.SetBounds(lx, 213, 140, 20);
		run.Controls.Add(keyTitle);
		keyBox = new TextBox();
		keyBox.Name = "openaikey";
		keyBox.SetBounds(cx, 210, cw, 24);
		keyBox.UseSystemPasswordChar = true;
		run.Controls.Add(keyBox);
		keyLabel = new Label();
		keyLabel.Name = "keynote";
		keyLabel.SetBounds(cx, 236, cw + 6, 18);
		keyLabel.ForeColor = Color.DimGray;
		keyLabel.Text = "Only for a model alone. Not stored.";
		run.Controls.Add(keyLabel);
		modeBox.SelectedIndexChanged += delegate
		{
			UpdateSearchLabel();
			UpdateButtons();
			if (UsesSearch && foulPlayOk == null && !fpChecking && !running) CheckFoulPlay();
			if (ModeKey == "guarded" && !running) SelectBestNetwork();
		};
		fpCheckButton = new Button();
		fpCheckButton.Name = "fpcheck";
		fpCheckButton.Text = "Check Foul Play";
		fpCheckButton.SetBounds(lx, 261, 130, 28);
		fpCheckButton.Click += delegate { CheckFoulPlay(); };
		run.Controls.Add(fpCheckButton);
		foulPlayLabel = new Label();
		foulPlayLabel.Name = "foulplay";
		foulPlayLabel.SetBounds(cx, 260, cw + 2, 36);
		foulPlayLabel.ForeColor = Color.DimGray;
		foulPlayLabel.Text = "Checking Foul Play in WSL ...";
		run.Controls.Add(foulPlayLabel);
		AddLabel(run, "Network", lx, 303);
		netBox = new ComboBox();
		netBox.Name = "network";
		netBox.DropDownStyle = ComboBoxStyle.DropDownList;
		netBox.SetBounds(cx, 300, cw - 60, 24);
		netBox.DropDownWidth = cw + 40;   // the open list shows each line whole
		netBox.DropDown += delegate { FillCheckpoints(true); };   // a network copied while the window is open shows up
		netBox.SelectionChangeCommitted += delegate { netPicked = true; };
		run.Controls.Add(netBox);
		// Unticked, the list holds the main line's newest networks and the baseline run's; ticked, every file.
		allNetsBox = new CheckBox();
		allNetsBox.Name = "allnets";
		allNetsBox.Text = "all";
		allNetsBox.SetBounds(cx + cw - 54, 301, 60, 22);
		allNetsBox.Checked = Get("allnets", "0") == "1";
		allNetsBox.CheckedChanged += delegate { FillCheckpoints(true); };
		new ToolTip().SetToolTip(allNetsBox, "Show every network file, the old ones too.");
		run.Controls.Add(allNetsBox);
		searchLabel = new Label();
		searchLabel.Name = "search";
		searchLabel.SetBounds(lx, 332, 404, 74);
		searchLabel.ForeColor = Color.DimGray;
		run.Controls.Add(searchLabel);
		ramLabel = new Label();
		ramLabel.Name = "ram";
		ramLabel.SetBounds(lx, 408, 404, 20);
		run.Controls.Add(ramLabel);
		UpdateSearchLabel();

		GroupBox status = statusBox = new GroupBox();
		status.Text = "Status";
		status.SetBounds(454, 8, 474, 434);
		status.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
		Controls.Add(status);
		stateLabel = new Label();
		stateLabel.Name = "state";
		stateLabel.SetBounds(14, 26, 446, 44);
		stateLabel.Font = new Font("Segoe UI", 10.5F, FontStyle.Bold);
		stateLabel.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
		status.Controls.Add(stateLabel);
		playedLabel = AddStatus(status, 76);
		playedLabel.Name = "played";
		ratingLabel = AddStatus(status, 102);
		ratingLabel.Name = "rating";
		progressLabel = AddStatus(status, 132);
		progressLabel.Name = "progress";
		progressLabel.Height = 128;
		lastLabel = AddStatus(status, 266);
		lastLabel.Name = "last";
		lastLabel.Height = 148;
		lastLabel.ForeColor = Color.DimGray;

		int by = 452;
		startButton = AddButton("Start", 12, by, 110);
		startButton.Name = "start";
		startButton.Font = new Font("Segoe UI", 9.5F, FontStyle.Bold);
		startButton.Click += delegate { StartRun(); };
		stopAfterButton = AddButton("Stop after these games", 130, by, 180);
		stopAfterButton.Name = "stopafter";
		stopAfterButton.Click += delegate { StopAfter(); };
		stopNowButton = AddButton("Stop now", 318, by, 110);
		stopNowButton.Name = "stopnow";
		stopNowButton.Click += delegate { AskStopNow(); };
		graphButton = AddButton("Elo graph", 454, by, 110);
		graphButton.Name = "graph";
		graphButton.Click += delegate { ShowGraph(); };
		Button folder = AddButton("Results folder", 572, by, 120);
		folder.Click += delegate { OpenFolder(); };

		logBox = new TextBox();
		logBox.Name = "log";
		logBox.Multiline = true;
		logBox.ReadOnly = true;
		logBox.ScrollBars = ScrollBars.Both;
		logBox.WordWrap = false;
		logBox.BackColor = Color.White;
		logBox.Font = new Font("Consolas", 9F);
		logBox.SetBounds(12, 492, 916, 292);
		logBox.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
		Controls.Add(logBox);

		// The column on the right: this run's Elo above, the network's neurons below, drawn here from
		// the run's "@app" lines. The bar between them moves (drag it) and so does the column's left
		// edge (the grip): both sizes are remembered (viewsplit, viewwidth).
		int vx = 940, vw = ViewColumn - 12;
		viewSplit = new SplitContainer();
		viewSplit.Name = "views";
		viewSplit.Orientation = Orientation.Horizontal;
		viewSplit.SetBounds(vx, 8, vw, 776);
		viewSplit.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Right;
		viewSplit.SplitterWidth = 7;
		viewSplit.BackColor = Color.FromArgb(200, 205, 212);
		Controls.Add(viewSplit);
		eloView = AddView("elo", viewSplit.Panel1, PaintElo);
		neuView = AddView("neurons", viewSplit.Panel2, PaintNeurons);
		new ToolTip().SetToolTip(viewSplit, "Drag the bar to give the Elo graph or the neurons more room.");
		viewGrip = new Panel();
		viewGrip.Name = "viewgrip";
		viewGrip.SetBounds(vx - 9, 8, 6, 776);
		viewGrip.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Right;
		viewGrip.BackColor = Color.FromArgb(200, 205, 212);
		viewGrip.Cursor = Cursors.SizeWE;
		new ToolTip().SetToolTip(viewGrip, "Drag to make the Elo and neuron panels wider or narrower.");
		int gripFrom = 0, widthFrom = 0;
		bool gripping = false;
		viewGrip.MouseDown += (s, e) => { gripping = true; gripFrom = Cursor.Position.X; widthFrom = viewSplit.Width; };
		viewGrip.MouseMove += (s, e) => { if (gripping) SetViewWidth(widthFrom + gripFrom - Cursor.Position.X); };
		viewGrip.MouseUp += (s, e) =>
		{
			if (!gripping) return;
			gripping = false;
			settings["viewwidth"] = ((int)(viewSplit.Width / ScaleK())).ToString(CultureInfo.InvariantCulture);
			SaveSettings();
		};
		Controls.Add(viewGrip);
		// The 3D blocks or the flat diagram: a "3D" pill drawn on the panel (PaintNeurons), clicked
		// to switch, remembered in the settings.
		view3d = Get("netview3d", "1") != "0";
		neuView.MouseClick += (s, e) =>
		{
			if (!togglePill.Contains(e.X, e.Y)) return;
			view3d = !view3d;
			settings["netview3d"] = view3d ? "1" : "0";
			SaveSettings();
			neuView.Invalidate();
		};
		neuView.MouseMove += (s, e) => { neuView.Cursor = togglePill.Contains(e.X, e.Y) ? Cursors.Hand : Cursors.Default; };
		new ToolTip().SetToolTip(neuView, "Click 3D to switch between the 3D blocks and the flat diagram.");
		netTick = new System.Windows.Forms.Timer();
		netTick.Interval = 33;
		netTick.Tick += delegate
		{
			neuView.Invalidate();
			if ((DateTime.Now - netAt).TotalMilliseconds > SweepMs + 100) netTick.Stop();
		};
		ShowStatus();

		AutoScaleDimensions = new SizeF(96F, 96F);
		AutoScaleMode = AutoScaleMode.Dpi;
		ResumeLayout(false);
		PerformLayout();
		float k = CurrentAutoScaleDimensions.Width / 96F;
		if (k < 1F) k = 1F;
		MinimumSize = new Size((int)((860 + ViewColumn) * k), (int)(660 * k));
		// The panels' sizes as they were left (after the scaling, so they are in this screen's pixels).
		viewSplit.Panel1MinSize = (int)(90 * k);
		viewSplit.Panel2MinSize = (int)(150 * k);
		int w = GetInt("viewwidth", 0);
		if (w > 0) SetViewWidth((int)(w * k));
		int split = GetInt("viewsplit", 240);
		viewSplit.SplitterDistance = Math.Max(viewSplit.Panel1MinSize, Math.Min(viewSplit.Height - viewSplit.Panel2MinSize - viewSplit.SplitterWidth, (int)(split * k)));
		viewSplit.SplitterMoved += delegate
		{
			settings["viewsplit"] = ((int)(viewSplit.SplitterDistance / ScaleK())).ToString(CultureInfo.InvariantCulture);
			SaveSettings();
		};
	}

	float ScaleK()
	{
		float k = CurrentAutoScaleDimensions.Width / 96F;
		return k < 1F ? 1F : k;
	}

	// The column's width (screen pixels), within what leaves the run and status boxes room; the
	// status box and the log end where the column's grip begins.
	void SetViewWidth(int width)
	{
		float k = ScaleK();
		int most = ClientSize.Width - (int)(640 * k), least = (int)(240 * k);
		width = Math.Max(least, Math.Min(Math.Max(least, most), width));
		int right = viewSplit.Right;
		viewSplit.SetBounds(right - width, viewSplit.Top, width, viewSplit.Height);
		viewGrip.Left = viewSplit.Left - viewGrip.Width - (int)(3 * k);
		int edge = viewGrip.Left - (int)(6 * k);
		statusBox.Width = Math.Max((int)(120 * k), edge - statusBox.Left);
		logBox.Width = Math.Max((int)(200 * k), edge - logBox.Left);
	}

	static decimal Clamp(int v, int lo, int hi) { return Math.Max(lo, Math.Min(hi, v)); }

	void AddLabel(Control parent, string text, int x, int y)
	{
		Label l = new Label();
		l.Text = text;
		l.SetBounds(x, y, 140, 20);
		parent.Controls.Add(l);
	}

	Label AddStatus(Control parent, int y)
	{
		Label l = new Label();
		l.SetBounds(14, y, 446, 22);
		l.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
		parent.Controls.Add(l);
		return l;
	}

	Button AddButton(string text, int x, int y, int w)
	{
		Button b = new Button();
		b.Text = text;
		b.SetBounds(x, y, w, 30);
		Controls.Add(b);
		return b;
	}

	int Helpers() { return Math.Min(MaxHelpers, (int)parallelBox.Value + 1); }

	// Who plays: the key kept in the settings file, and the list's text. The order is the list's.
	// The bot (the network, with or without the search), or one model alone with no network and
	// no search behind it. The modes where a model had the last word over the bot, and reading
	// ahead, are gone from the window: on the ladder neither played better than the bot itself.
	static readonly string[] ModeKeys = { "lookahead", "guarded", "net", "fable", "opus", "sonnet", "haiku", "solalone", "lunaalone" };
	static readonly string[] ModeTexts = {
		"Bot: network + search (Lookahead)", "Bot: best network + guarded search (new)", "Bot: network only",
		"Claude Fable 5.1 alone", "Claude Opus 5.5 alone", "Claude Sonnet 5 alone", "Claude Haiku 4.5 alone",
		"GPT-6 Sol alone (OpenAI)", "GPT-6 Luna alone (OpenAI)" };
	static readonly string[] EffortKeys = { "low", "medium", "high" };
	static readonly string[] EffortTexts = {
		"Thinking: low (fastest)", "Thinking: medium", "Thinking: high (slowest)" };

	// A mode saved by an earlier version of the window.
	static string SavedMode(string saved)
	{
		if (saved == "lunasolo") return "lunaalone";
		if (saved == "solsolo") return "solalone";
		if (saved == "luna" || saved == "sol") return "lookahead";
		return saved;
	}

	string ModeKey { get { return modeBox == null || modeBox.SelectedIndex < 0 ? "lookahead" : ModeKeys[modeBox.SelectedIndex]; } }
	string EffortKey { get { return effortBox == null || effortBox.SelectedIndex < 0 ? "low" : EffortKeys[effortBox.SelectedIndex]; } }
	static bool IsClaude(string mode) { return mode == "fable" || mode == "opus" || mode == "sonnet" || mode == "haiku"; }
	static bool IsSolo(string mode) { return IsClaude(mode) || mode == "solalone" || mode == "lunaalone"; }
	static bool IsModel(string mode) { return IsSolo(mode); }
	bool UsesSearch { get { return ModeKey == "lookahead" || ModeKey == "guarded"; } }   // a model alone starts no search helpers
	bool UsesModel { get { return IsModel(ModeKey); } }
	bool UsesNetwork { get { return !IsSolo(ModeKey); } }

	// Whose key a model needs: the variable an empty key box stands for, and the company's name.
	static string KeyEnv(string mode) { return IsClaude(mode) ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"; }
	static string KeyOwner(string mode) { return IsClaude(mode) ? "Anthropic" : "OpenAI"; }

	static string ModelName(string mode)
	{
		if (mode == "fable") return "Claude Fable 5.1";
		if (mode == "opus") return "Claude Opus 5.5";
		if (mode == "sonnet") return "Claude Sonnet 5";
		if (mode == "haiku") return "Claude Haiku 4.5";
		if (mode == "solalone") return "Sol";
		if (mode == "lunaalone") return "Luna";
		return "";
	}

	// ladder.py's --player for a mode (the second argument is from when the window could read ahead).
	static string PlayerOf(string mode, bool ahead)
	{
		if (mode == "net") return "net";
		if (IsClaude(mode)) return mode + "-solo";
		if (mode == "solalone") return "sol-solo";
		if (mode == "lunaalone") return "luna-solo";
		if (mode == "guarded") return "mix-trust";   // the search may only pick moves the network gives 3% or more (reports/2026-10-15/LOSS_STUDY.md)
		return "mix";
	}

	static string ModeNote(string mode, bool ahead)
	{
		if (mode == "net") return "Network only";
		if (mode == "guarded") return "Guarded search";
		if (IsSolo(mode)) return ModelName(mode) + " alone";
		return "Lookahead";
	}

	void UpdateSearchLabel()
	{
		// The time the plan expects, as the search line says it: Wider moves in whole teams (2 more
		// guessed teams, about 1.1 s), so neighbouring stops can give the same plan and the same time.
		if (thinkLabel != null)
			thinkLabel.Text = "Thinking: about\r\n" + Thinking.Tenths(Thinking.Plan(ThinkSeconds, WiderPicked, "mix").AboutSeconds) + " s a move";
		if (searchLabel == null || parallelBox == null || modeBox == null) return;
		if (IsSolo(ModeKey))
		{
			string m = ModeKey;
			string cost = m == "fable" ? "about 5 cents a move, $2 a game"
				: m == "opus" ? "about 2 cents a move, under $1 a game"
				: m == "sonnet" ? "about 1 cent a move"
				: m == "haiku" ? "1 to 2 s and under half a cent a move"
				: m == "solalone" ? "2 to 3 s and half a cent a move"
				: "0.3 s and a hundredth of a cent a move";
			searchLabel.Text = ModelName(m) + " alone picks every move from the position: " + cost + "." +
				(IsClaude(m) && m != "haiku" ? " A slow answer is waited for up to 25 s, then a simple rule plays." : "") +
				" No network, no search. Stops at $20 of API use. Expect it far below Lookahead.";
			return;
		}
		if (!UsesSearch)
		{
			searchLabel.Text = "Network only: the network's best move, no search. Network on the GPU.";
			return;
		}
		string text = Thinking.Plan(ThinkSeconds, WiderPicked, PlayerOf(ModeKey, false)).Describe();
		if (ModeKey == "guarded")
			text = "Guarded (overrules only with moves the network gives 3% or more). " + text;
		if (lastThinkingLine.Length > 0) text += "\r\n" + lastThinkingLine;
		searchLabel.Text = text;
	}

	void UpdateRam()
	{
		double gb = Memory.FreeGB();
		if (gb < 0) { ramLabel.Text = "Free memory: unknown"; return; }
		ramLabel.Text = "Free memory: " + gb.ToString("0.0", CultureInfo.InvariantCulture) + " GB" +
			(gb < LowMemoryGB ? "  (low: " + LowMemoryGB.ToString("0", CultureInfo.InvariantCulture) + " GB or more is advised)" : "");
		ramLabel.ForeColor = gb < LowMemoryGB ? Color.Firebrick : Color.Black;
	}

	void UpdateButtons()
	{
		bool idle = !running;
		startButton.Enabled = idle && !closeWhenDone;
		stopAfterButton.Enabled = running && !stoppingAfter && !stoppingNow;
		stopNowButton.Enabled = running && !stoppingNow;
		graphButton.Enabled = !graphRunning;
		userBox.Enabled = passBox.Enabled = gamesBox.Enabled = parallelBox.Enabled = netBox.Enabled = idle;
		modeBox.Enabled = idle;   // a failed check never changes the choice (Start asks)
		netBox.Enabled = allNetsBox.Enabled = idle && UsesNetwork;   // a model alone plays without a network
		effortBox.Enabled = idle && IsClaude(ModeKey) && ModeKey != "haiku";
		thinkBar.Visible = thinkLabel.Visible = styleBox.Visible = UsesSearch;
		effortBox.Visible = !UsesSearch;
		thinkBar.Enabled = styleBox.Enabled = idle && UsesSearch;
		keyTitle.Visible = keyBox.Visible = keyLabel.Visible = !UsesSearch;   // a search mode uses no key: its row holds the search style
		keyBox.Enabled = idle && UsesModel;
		keyTitle.Text = UsesModel ? KeyOwner(ModeKey) + " API key" : "API key";
		keyLabel.Text = UsesModel ? "Not stored. Empty: " + KeyEnv(ModeKey) : "Only for a model alone. Not stored.";
		fpCheckButton.Enabled = idle && !fpChecking;
	}

	void ShowStatus()
	{
		string state;
		if (!running) state = closeWhenDone ? "Closing" : "Not running";
		else if (stoppingNow) state = "Stopping now";
		else if (stoppingAfter) state = closeWhenDone ? "Finishing the games in progress, then closing" : "Stopping after the games in progress";
		else if (!sawStart) state = "Starting" + (phase.Length > 0 ? ": " + phase : " ...");
		else if (games.Count == 0) state = "Running: looking for an opponent";
		else state = "Running";
		stateLabel.Text = state;
		stateLabel.ForeColor = running ? (stoppingAfter || stoppingNow ? Color.DarkOrange : Color.ForestGreen) : Color.DimGray;
		int played = wins + losses;
		playedLabel.Text = "Games played: " + played + (gamesTarget > 0 ? " of " + gamesTarget : "") + "     Record: " + wins + " - " + losses +
			(played > 0 ? "  (" + (100 * wins / played) + "% won)" : "");
		string r = "Rating: at the start " + (ratingStart.HasValue ? ratingStart.Value.ToString() : "?") +
			",  now " + (ratingNow.HasValue ? ratingNow.Value.ToString() : "?");
		if (ratingStart.HasValue && ratingNow.HasValue)
		{
			int d = ratingNow.Value - ratingStart.Value;
			r += "  (" + (d >= 0 ? "+" : "") + d + ")";
		}
		ratingLabel.Text = r;
		if (games.Count == 0) progressLabel.Text = "Games in progress: none";
		else
		{
			List<GameInfo> list = new List<GameInfo>(games.Values);
			list.Sort((a, b) => a.N.CompareTo(b.N));
			StringBuilder sb = new StringBuilder("Games in progress:");
			foreach (GameInfo g in list)
				sb.Append("\n   game " + g.N + " vs " + (g.Opponent ?? "?") + (g.Turn > 0 ? ", turn " + g.Turn : ", starting"));
			progressLabel.Text = sb.ToString();
		}
	}

	// --------------------------------------------------------------------------------------
	// Settings: the username and the run's numbers. Never the password.
	// --------------------------------------------------------------------------------------

	void LoadSettings()
	{
		try
		{
			if (!File.Exists(settingsFile)) return;
			foreach (string line in File.ReadAllLines(settingsFile, Encoding.UTF8))
			{
				int eq = line.IndexOf('=');
				if (eq > 0) settings[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
			}
			if (settings.ContainsKey("python") && settings["python"].Length > 0) pythonExe = settings["python"];
			if (settings.ContainsKey("checkpoints") && settings["checkpoints"].Length > 0) checkpointsDir = settings["checkpoints"];
			if (settings.ContainsKey("armc") && settings["armc"].Length > 0) armcDir = settings["armc"];
			if (settings.ContainsKey("v2") && settings["v2"].Length > 0) v2Dir = settings["v2"];
		}
		catch (Exception) { }
	}

	string Get(string key, string dflt) { string v; return settings.TryGetValue(key, out v) ? v : dflt; }
	int GetInt(string key, int dflt) { int v; return int.TryParse(Get(key, ""), out v) ? v : dflt; }

	void SaveSettings()
	{
		settings["user"] = userBox.Text.Trim();
		settings["games"] = ((int)gamesBox.Value).ToString();
		settings["parallel"] = ((int)parallelBox.Value).ToString();
		settings["mode"] = ModeKey;
		settings["effort"] = EffortKey;
		settings["think"] = Thinking.Seconds(ThinkSeconds);
		settings["searchstyle"] = WiderPicked ? "wider" : "deeper";
		settings["netview3d"] = view3d ? "1" : "0";
		settings["allnets"] = allNetsBox.Checked ? "1" : "0";
		try
		{
			StringBuilder sb = new StringBuilder("# " + Program.AppName + " settings (no password is ever kept here)\r\n");
			foreach (KeyValuePair<string, string> kv in settings)
				if (kv.Key != "password" && kv.Key != "key" && kv.Key != "openaikey") sb.Append(kv.Key + "=" + kv.Value + "\r\n");
			File.WriteAllText(settingsFile, sb.ToString(), new UTF8Encoding(false));
		}
		catch (Exception e) { Log("could not save the settings: " + e.Message); }
	}

	// --------------------------------------------------------------------------------------
	// The networks
	// --------------------------------------------------------------------------------------

	// Lists the networks again. keepPicked: a network picked by hand stays selected (when it is
	// still there); otherwise the newest whole one that is not still being written is selected.
	// The best network (the main run's final update 24381: first in every rating fit of 2026-10-17 to
	// 10-20, docs/DETAILS.md). The main run's update 11850 played the ladder peak of 2,466.
	// Always in the list, marked; the guarded mode picks it. A "best=" line in the settings names another file.
	bool IsBest(Checkpoint c)
	{
		return c != null && !c.ArmC && !c.Baseline && c.V2Arm == null &&
			string.Equals(System.IO.Path.GetFileName(c.Path), Get("best", "learner_update_024381.pt"), StringComparison.OrdinalIgnoreCase);
	}

	void SelectBestNetwork()
	{
		for (int i = 0; i < netBox.Items.Count; i++)
		{
			Checkpoint c = netBox.Items[i] as Checkpoint;
			if (IsBest(c) && c.Complete) { netBox.SelectedIndex = i; netPicked = true; return; }
		}
		Log("the best network (" + Get("best", "learner_update_024381.pt") + ") is not in " + checkpointsDir + ": pick one in the list");
	}

	void FillCheckpoints(bool keepPicked)
	{
		Checkpoint before = netBox.SelectedItem as Checkpoint;
		string keep = keepPicked && netPicked && before != null ? before.Path : null;
		List<Checkpoint> found = new List<Checkpoint>();
		try
		{
			// learner_update_N.pt: the main line. baseline_update_N.pt: the baseline run of the search
			// branch, kept under its own name because its update numbers overlap the main line's.
			foreach (string f in Directory.GetFiles(checkpointsDir, "*_update_*.pt"))
			{
				Match m = Regex.Match(System.IO.Path.GetFileName(f), @"^(learner|baseline)_update_(\d+)\.pt$");
				if (!m.Success) continue;
				Checkpoint c = new Checkpoint();
				c.Path = f;
				c.Baseline = m.Groups[1].Value == "baseline";
				c.Update = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
				c.Written = File.GetLastWriteTime(f);
				c.Fresh = (DateTime.Now - c.Written).TotalMinutes < 2;
				c.Complete = Checkpoint.LooksComplete(f);
				found.Add(c);
			}
		}
		catch (Exception e) { Log("could not list the networks in " + checkpointsDir + ": " + e.Message); }
		// armC-full_update_N_xM.pt: the full run from arm C, in its own folder (its numbers overlap both).
		try
		{
			if (Directory.Exists(armcDir))
				foreach (string f in Directory.GetFiles(armcDir, "armC-full_update_*.pt"))
				{
					Match m = Regex.Match(System.IO.Path.GetFileName(f), @"^armC-full_update_(\d+)_(\d+)M\.pt$");
					if (!m.Success) continue;
					Checkpoint c = new Checkpoint();
					c.Path = f;
					c.ArmC = true;
					c.Update = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
					c.Millions = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
					c.Written = File.GetLastWriteTime(f);
					c.Fresh = (DateTime.Now - c.Written).TotalMinutes < 2;
					c.Complete = Checkpoint.LooksComplete(f);
					found.Add(c);
				}
		}
		catch (Exception e) { Log("could not list the networks in " + armcDir + ": " + e.Message); }
		// v2-<arm>_update_N_xM.pt: the v2 training waves, in their own folder (network only, exported).
		try
		{
			if (Directory.Exists(v2Dir))
				foreach (string f in Directory.GetFiles(v2Dir, "v2-*_update_*.pt"))
				{
					Match m = Regex.Match(System.IO.Path.GetFileName(f), @"^v2-([A-Za-z0-9]+)_update_(\d+)_(\d+)M\.pt$");
					if (!m.Success) continue;
					Checkpoint c = new Checkpoint();
					c.Path = f;
					c.V2Arm = m.Groups[1].Value;
					c.Update = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
					c.Millions = int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);
					c.Written = File.GetLastWriteTime(f);
					c.Fresh = (DateTime.Now - c.Written).TotalMinutes < 2;
					c.Complete = Checkpoint.LooksComplete(f);
					found.Add(c);
				}
		}
		catch (Exception e) { Log("could not list the networks in " + v2Dir + ": " + e.Message); }
		// The main line first, so that what is selected by default stays its newest network.
		// Newest saved first, whichever run it is from: the top of the list is the latest network.
		found.Sort((a, b) => b.Written != a.Written ? b.Written.CompareTo(a.Written) : b.Update.CompareTo(a.Update));
		if (allNetsBox != null && !allNetsBox.Checked)
		{
			// Without "all": the main line's newest three, the baseline run's and arm C's. One picked by hand stays.
			List<Checkpoint> shown = new List<Checkpoint>();
			int main = 0;
			foreach (Checkpoint c in found)
				if (c.Line > 0 || c.Path == keep || IsBest(c) || main++ < 3) shown.Add(c);
			found = shown;
		}
		netBox.BeginUpdate();
		netBox.Items.Clear();
		int select = -1, kept = -1, whole = -1;
		for (int i = 0; i < found.Count; i++)
		{
			Checkpoint c = found[i];
			// Plain words: which training run, how much it trained, when it was saved; the newest says LATEST.
			string v2Name;
			string run = c.V2Arm != null ? "NEW TRAINING: " + (V2Names.TryGetValue(c.V2Arm, out v2Name) ? v2Name : c.V2Arm) + ", " + c.Millions + "M new moves"
				: c.ArmC ? "Top-players run, " + c.Millions + "M moves trained" : c.Baseline ? "Baseline run" : "Main run";
			c.Label = (i == 0 ? "LATEST   " : IsBest(c) ? "BEST ON LADDER   " : "") + run + ", saved " + c.Written.ToString("d MMM HH:mm", CultureInfo.InvariantCulture) +
				(!c.Complete ? (c.Fresh ? "   (still copying)" : "   (broken file, can't use)") : c.Fresh ? "   (just copied)" : "");
			netBox.Items.Add(c);
			if (select < 0 && c.Complete && !c.Fresh) select = i;
			if (whole < 0 && c.Complete) whole = i;
			if (keep != null && c.Path == keep) kept = i;
		}
		netBox.EndUpdate();
		if (found.Count == 0) { Log("no networks (learner_update_*.pt or baseline_update_*.pt) in " + checkpointsDir + ", nor armC-full_update_*.pt in " + armcDir); return; }
		if (kept >= 0) netBox.SelectedIndex = kept;
		else
		{
			netPicked = false;
			netBox.SelectedIndex = select >= 0 ? select : whole >= 0 ? whole : 0;
		}
	}

	// --------------------------------------------------------------------------------------
	// Foul Play in WSL
	// --------------------------------------------------------------------------------------

	// On opening: checked only for Lookahead and with enough free memory (the check starts Python
	// and, when it is not up, the WSL machine). Otherwise "Check Foul Play" or Start checks it.
	void FirstFoulPlayCheck()
	{
		double gb = Memory.FreeGB();
		if (!UsesSearch)
		{
			foulPlayLabel.Text = "Foul Play in WSL: not checked (this choice plays without the search)";
			foulPlayLabel.ForeColor = Color.DimGray;
		}
		else if (gb >= 0 && gb < LowMemoryGB)
		{
			foulPlayLabel.Text = "Foul Play in WSL: not checked yet, memory is low (Check Foul Play, or Start checks it)";
			foulPlayLabel.ForeColor = Color.DimGray;
		}
		else CheckFoulPlay();
	}

	void CheckFoulPlay()
	{
		if (fpChecking || running) return;
		fpChecking = true;
		foulPlayOk = null;
		foulPlayLabel.Text = "Checking Foul Play in WSL ...";
		foulPlayLabel.ForeColor = Color.DimGray;
		UpdateButtons();
		string code = "import foulplay_bridge as fb\nok, why = fb.available()\nprint(('OK ' if ok else 'NO ') + why.replace('\\n', ' '))\n";
		Thread t = new Thread(delegate ()
		{
			bool ok = false;
			string why = "the check did not run";
			try
			{
				string output = RunPythonQuick(new string[] { "-c", code }, null, 240000);
				string line = null;
				foreach (string l in output.Split('\n')) if (l.StartsWith("OK ") || l.StartsWith("NO ")) line = l.Trim();
				if (line != null) { ok = line.StartsWith("OK "); why = line.Substring(3); }
				else why = "the check gave no answer: " + Shorten(output.Trim(), 200);
			}
			catch (Exception e) { why = e.Message; }
			try { BeginInvoke((MethodInvoker)delegate { FoulPlayChecked(ok, why); }); } catch (Exception) { }   // the window may be gone
		});
		t.IsBackground = true;
		t.Start();
	}

	void FoulPlayChecked(bool ok, string why)
	{
		fpChecking = false;
		foulPlayOk = ok;
		foulPlayWhy = why;
		if (ok)
		{
			Match m = Regex.Match(why, @"foul-play ([0-9a-f]{7})");
			foulPlayLabel.Text = "Foul Play in WSL: ready" + (m.Success ? " (" + m.Groups[1].Value + ")" : "");
			foulPlayLabel.ForeColor = Color.ForestGreen;
		}
		else
		{
			// The choice stays as it is: Start asks whether to play the network alone this once.
			foulPlayLabel.Text = "Lookahead unavailable: " + Shorten(why, 100);
			foulPlayLabel.ForeColor = Color.Firebrick;
			Log("Lookahead is unavailable: " + why + "   (Check Foul Play tries again)");
		}
		new ToolTip().SetToolTip(foulPlayLabel, why);
		UpdateButtons();
	}

	static string Shorten(string s, int n) { return s.Length <= n ? s : s.Substring(0, n - 3) + "..."; }

	// Runs the venv's Python next to this exe, hidden, in a job of its own; its output (both streams).
	string RunPythonQuick(string[] args, Dictionary<string, string> env, int timeoutMs)
	{
		ProcessStartInfo psi = PythonStart(args);
		psi.RedirectStandardOutput = psi.RedirectStandardError = true;
		if (env != null) foreach (KeyValuePair<string, string> kv in env) psi.EnvironmentVariables[kv.Key] = kv.Value;
		StringBuilder output = new StringBuilder();
		using (Job job = new Job())
		using (Process p = new Process())
		{
			p.StartInfo = psi;
			p.OutputDataReceived += (s, e) => { if (e.Data != null) lock (output) output.Append(e.Data).Append('\n'); };
			p.ErrorDataReceived += (s, e) => { if (e.Data != null) lock (output) output.Append(e.Data).Append('\n'); };
			p.Start();
			try { job.Add(p); } catch (Exception) { }
			p.StandardInput.Close();
			p.BeginOutputReadLine();
			p.BeginErrorReadLine();
			if (!p.WaitForExit(timeoutMs)) { job.Kill(); throw new Exception("no answer within " + (timeoutMs / 1000) + " s"); }
			p.WaitForExit();
		}
		lock (output) return output.ToString();
	}

	ProcessStartInfo PythonStart(string[] args)
	{
		ProcessStartInfo psi = new ProcessStartInfo(pythonExe, JoinArgs(args));
		psi.UseShellExecute = false;
		psi.CreateNoWindow = true;
		psi.WorkingDirectory = appDir;
		psi.RedirectStandardInput = true;
		psi.StandardOutputEncoding = Encoding.UTF8;
		psi.StandardErrorEncoding = Encoding.UTF8;
		psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
		psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
		psi.EnvironmentVariables["PYTHONWARNINGS"] = "ignore:flex_attention called without torch.compile";
		psi.EnvironmentVariables.Remove("PS_PASSWORD");   // none from this app's own environment: the password goes on stdin
		return psi;
	}

	// Windows command-line quoting (the rules Python's argv parsing follows).
	static string JoinArgs(string[] args)
	{
		StringBuilder sb = new StringBuilder();
		foreach (string a in args)
		{
			if (sb.Length > 0) sb.Append(' ');
			if (a.Length > 0 && a.IndexOfAny(new char[] { ' ', '\t', '"', '\n' }) < 0) { sb.Append(a); continue; }
			sb.Append('"');
			int slashes = 0;
			foreach (char ch in a)
			{
				if (ch == '\\') { slashes++; continue; }
				if (ch == '"') { sb.Append('\\', slashes * 2 + 1); sb.Append('"'); }
				else { sb.Append('\\', slashes); sb.Append(ch); }
				slashes = 0;
			}
			sb.Append('\\', slashes * 2);
			sb.Append('"');
		}
		return sb.ToString();
	}

	// --------------------------------------------------------------------------------------
	// Start and stop
	// --------------------------------------------------------------------------------------

	void StartRun()
	{
		if (running) return;
		string user = userBox.Text.Trim();
		if (user.Length == 0) { Warn("Type the Showdown username first."); return; }
		if (!opts.Test && passBox.Text.Length == 0) { Warn("Type the password of " + user + " first. It is not stored."); return; }
		if (!File.Exists(pythonExe)) { Warn("Python was not found at " + pythonExe + "."); return; }
		if (!File.Exists(System.IO.Path.Combine(appDir, "ladder.py"))) { Warn("ladder.py was not found next to this program, in " + appDir + "."); return; }

		// One bot per account: not next to another ladder.py playing on the same server.
		string server = ServerArg();
		List<OtherBot> others = null;
		Cursor = Cursors.WaitCursor;
		try { others = OtherBot.Find(0); }
		catch (Exception e) { Log("could not look for another bot on this computer (" + e.Message + "); ladder.py's own lock still refuses a second run on the same account"); }
		finally { Cursor = Cursors.Default; }
		if (others != null)
		{
			foreach (OtherBot o in others)
			{
				if (!o.Clashes(server, user)) continue;
				if (o.User != null)
				{
					Warn("A bot is already playing as " + o.User + " on " + ServerText(server) + " (process " + o.Pid + ", started " + o.Started +
						").\n\nTwo bots on one account fight over the same games. Stop that one first.");
					return;
				}
				string ask = "Another bot is already playing on " + ServerText(server) + " (process " + o.Pid + ", started " + o.Started +
					"; probably the Desktop's Look Play.cmd). Its account is not on its command line.\n\n" +
					"Two bots on one account fight over the same games. Stop it first (Look Stop.cmd, or close its window).\n\n" +
					"Start anyway? Only if it plays a different account than " + user + ".";
				if (MessageBox.Show(this, ask, Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
				break;
			}
		}

		// The networks again: one copied since the window opened, and whether the file is whole.
		FillCheckpoints(true);
		Checkpoint ckpt = netBox.SelectedItem as Checkpoint;
		if (ckpt == null) { Warn("No network is selected (none found in " + checkpointsDir + ")."); return; }
		if (!ckpt.Complete)
		{
			Warn("The network file " + System.IO.Path.GetFileName(ckpt.Path) + " is incomplete (" + (ckpt.Fresh ? "it is still being copied" : "an interrupted copy?") +
				"). Pick another network.");
			return;
		}
		if (ckpt.Fresh && MessageBox.Show(this, "The network " + System.IO.Path.GetFileName(ckpt.Path) + " was written less than two minutes ago. It looks whole, but it may still be being copied.\n\nUse it anyway?",
			Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;

		string mode = ModeKey;
		bool ahead = false;
		bool look = UsesSearch;
		string lookNote = "";
		if (look && foulPlayOk == false)
		{
			string ask = "Lookahead is unavailable: " + foulPlayWhy + "\n\nPlay with the network only this time?\n\n" +
				"(The choice in the list stays. \"Check Foul Play\" tries again.)";
			if (MessageBox.Show(this, ask, Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
			look = false;
			mode = "net";
			ahead = false;
			lookNote = " (Lookahead was unavailable)";
		}
		bool model = IsModel(mode);
		if (model && keyBox.Text.Trim().Length == 0 && string.IsNullOrEmpty(Environment.GetEnvironmentVariable(KeyEnv(mode))))
		{
			Warn("Type the " + KeyOwner(mode) + " API key first (" + ModelName(mode) + " is a model of " + KeyOwner(mode) + "). It is not stored.");
			return;
		}
		if (IsSolo(mode))
		{
			string alone = ModeNote(mode, false) + " plays RATED games as " + user + " with no network and no search behind it.\n\n" +
				"Expect it to play far below Lookahead: this account's rating can fall a long way. " +
				"A second account shows what it can do without that risk.\n\nStart on " + user + "?";
			if (MessageBox.Show(this, alone, Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
		}
		if (model && keyBox.Text.IndexOfAny(new char[] { '\r', '\n' }) >= 0)
		{
			Warn("The " + KeyOwner(mode) + " API key has a line break in it. Paste it again.");
			return;
		}
		double gb = Memory.FreeGB();
		if (gb >= 0 && gb < LowMemoryGB)
		{
			string msg = "Only " + gb.ToString("0.0", CultureInfo.InvariantCulture) + " GB of memory is free (" +
				LowMemoryGB.ToString("0", CultureInfo.InvariantCulture) + " GB or more is advised: the network, " +
				(look ? Helpers() + " search helpers in WSL " : "") + "and the games need it).\n\nStart anyway?";
			if (MessageBox.Show(this, msg, Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
		}
		SaveSettings();

		Directory.CreateDirectory(runsDir);
		string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
		stopFile = System.IO.Path.Combine(runsDir, "stop-" + stamp);
		stopNowFile = System.IO.Path.Combine(runsDir, "stopnow-" + stamp);
		int parallel = (int)parallelBox.Value;
		gamesTarget = (int)gamesBox.Value;
		List<string> args = new List<string> {
			"-u", "ladder.py", "--app-events", "--password-stdin",
			"--games", gamesTarget.ToString(), "--parallel", parallel.ToString(),
			"--checkpoint", ckpt.Path, "--user", user,
			"--stop-file", stopFile, "--stop-now-file", stopNowFile,
			"--server", server,
		};
		if (opts.Test) { args.Add("--out"); args.Add(System.IO.Path.Combine(appDir, "runs", "ladder-test", "results.jsonl")); }
		ThinkPlan plan = look ? Thinking.Plan(ThinkSeconds, WiderPicked, PlayerOf(mode, ahead)) : null;
		if (look)
		{
			args.AddRange(plan.Args(Helpers()));   // the slider and the search style (Thinking.Plan)
			if (model) args.Add("--key-stdin");   // the key is the second line of the child's standard input
		}
		else
		{
			// Network only, or a model alone: no search is started.
			args.Add("--player"); args.Add(PlayerOf(mode, false));
			if (model) args.Add("--key-stdin");
			if (IsClaude(mode) && mode != "haiku") { args.Add("--claude-effort"); args.Add(EffortKey); }
		}
		// The network panels: what it looks at and which neurons fire, for every move it makes.
		if (!IsSolo(mode)) args.Add("--app-net-view");

		ProcessStartInfo psi = PythonStart(args.ToArray());   // no PS_PASSWORD in it: the password goes on stdin
		psi.RedirectStandardOutput = psi.RedirectStandardError = true;
		if (!string.IsNullOrEmpty(opts.TestFoulPlayDir)) psi.EnvironmentVariables["FOULPLAY_DIR"] = opts.TestFoulPlayDir;

		// Reset the status of the last run.
		wins = losses = abandoned = 0;
		ratingStart = ratingNow = null;
		eloTrail.Clear();
		eloWon.Clear();
		netNow = null;
		followTag = null;
		neuScale = null;
		runUsesNet = !IsSolo(mode);
		eloView.Invalidate(); neuView.Invalidate();
		games.Clear();
		gameNumbers.Clear();
		lastStderr.Clear();
		lastError = null;
		phase = "";
		sawStart = sawFinished = stoppingAfter = stoppingNow = false;
		lastLabel.Text = "";
		try
		{
			runLog = new StreamWriter(System.IO.Path.Combine(runsDir, "run-" + stamp + ".log"), false, new UTF8Encoding(false));
			runLog.AutoFlush = true;
		}
		catch (Exception) { runLog = null; }
		Log("---- " + DateTime.Now.ToString("dd MMM yyyy HH:mm:ss", CultureInfo.InvariantCulture) + "  start: " + user + ", " +
			gamesTarget + " games, " + parallel + " at once, " + (look ? ModeNote(mode, ahead) + " (" + plan.LogText() + ", " + Helpers() + " helpers)" : ModeNote(mode, false) + lookNote) +
			", " + System.IO.Path.GetFileName(ckpt.Path) + (opts.Test ? ", TEST server 127.0.0.1:" + opts.TestPort : ", play.pokemonshowdown.com"));
		Log("command: " + System.IO.Path.GetFileName(pythonExe) + " " + JoinArgs(args.ToArray()) + "   (the password" +
			(model ? " and the " + KeyOwner(mode) + " key go" : " goes") + " on its standard input, nowhere else)");

		Process p = new Process();
		p.StartInfo = psi;
		p.EnableRaisingEvents = true;
		outDone.Reset();
		errDone.Reset();
		p.OutputDataReceived += (s, e) => { if (e.Data == null) outDone.Set(); else OnChildLine(e.Data, false); };
		p.ErrorDataReceived += (s, e) => { if (e.Data == null) errDone.Set(); else OnChildLine(e.Data, true); };
		p.Exited += delegate
		{
			// The streams end after the process: wait for their last lines (a few seconds at most).
			WaitHandle.WaitAll(new WaitHandle[] { outDone, errDone }, 10000);
			try { BeginInvoke((MethodInvoker)OnChildExit); } catch (Exception) { }
		};
		Job job = new Job();
		try
		{
			p.Start();
		}
		catch (Exception e)
		{
			job.Dispose();
			outDone.Set(); errDone.Set();
			Log("could not start Python: " + e.Message);
			CloseRunLog();
			return;
		}
		try { job.Add(p); }
		catch (Exception e) { Log("warning: the bot could not be tied to this window (" + e.Message + "); close it with Stop, not by ending this program"); }
		child = p;
		childJob = job;
		running = true;
		runServer = server;
		runUser = user;
		scanTicks = 0;
		// The password: one line of UTF-8 on the child's standard input, which is then closed. The
		// venv's python.exe hands that pipe to the interpreter it starts, so the password is in no
		// environment block, no command line and no file (ladder.py --password-stdin).
		try
		{
			// With a model the OpenAI key is the second line (ladder.py --key-stdin); an empty one
			// means the variable OPENAI_API_KEY, which the child has when this computer has it.
			byte[] line = new UTF8Encoding(false).GetBytes(passBox.Text + "\n" + (model ? keyBox.Text.Trim() + "\n" : ""));
			try
			{
				Stream input = p.StandardInput.BaseStream;
				input.Write(line, 0, line.Length);
				input.Flush();
			}
			finally { Array.Clear(line, 0, line.Length); }
		}
		catch (Exception e) { Log("could not hand the password" + (model ? " and the key" : "") + " to the bot: " + e.Message); }
		try { p.StandardInput.Close(); } catch (Exception) { }
		p.BeginOutputReadLine();
		p.BeginErrorReadLine();
		UpdateButtons();
		ShowStatus();
	}

	void StopAfter()
	{
		if (!running || stoppingAfter || stoppingNow) return;
		try { File.WriteAllText(stopFile, "stop after the games in progress\r\n"); }
		catch (Exception e) { Warn("Could not write the stop file " + stopFile + ": " + e.Message); return; }
		stoppingAfter = true;
		Log("stop asked: no new games, the ones in progress are finished");
		UpdateButtons();
		ShowStatus();
	}

	void AskStopNow()
	{
		if (!running) return;
		int playing = games.Count;
		string msg = playing > 0
			? "Stop now? The " + (playing == 1 ? "game" : playing + " games") + " in progress will be lost on time.\n\n\"Stop after these games\" finishes them first."
			: "Stop now?";
		if (MessageBox.Show(this, msg, Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
		StopNow();
	}

	void StopNow()
	{
		if (!running || stoppingNow) return;
		stoppingNow = true;
		stopNowAt = DateTime.Now;
		try { File.WriteAllText(stopNowFile, "stop now\r\n"); }
		catch (Exception) { }
		Log("stop now asked: the bot stops at once and stops its search helpers");
		UpdateButtons();
		ShowStatus();
	}

	// A bot that has not stopped 20 s after "stop now" (it may be loading the network or starting
	// its helpers, when it does not look for the file) is ended with everything it started.
	void CheckStopNowTimeout()
	{
		if (running && stoppingNow && childJob != null && (DateTime.Now - stopNowAt).TotalSeconds > 20)
		{
			Log("the bot did not stop within 20 s: ending it and everything it started");
			childJob.Kill();
			stopNowAt = DateTime.MaxValue.AddDays(-1);
		}
	}

	// Every 15 s during a run: another bot that started playing on the same server (the Desktop's
	// Look Play.cmd does not take ladder.py's lock) is reported, once, in red.
	void ScanDuringRun()
	{
		if (!running || scanning || child == null || ++scanTicks % 5 != 0) return;
		int own;
		try { own = child.Id; } catch (Exception) { return; }
		string server = runServer, user = runUser;
		scanning = true;
		ThreadPool.QueueUserWorkItem(delegate
		{
			List<OtherBot> others = null;
			try { others = OtherBot.Find(own); } catch (Exception) { }
			try
			{
				BeginInvoke((MethodInvoker)delegate
				{
					scanning = false;
					if (others == null || !running) return;
					foreach (OtherBot o in others)
					{
						if (!o.Clashes(server, user) || warnedBots.Contains(o.Pid)) continue;
						warnedBots.Add(o.Pid);
						string w = "WARNING: another bot is playing on " + ServerText(server) + " (process " + o.Pid + ", started " + o.Started +
							(o.User != null ? ", as " + o.User : "") + "). Two bots on one account fight over the same games: stop one of them.";
						Log(w);
						lastLabel.ForeColor = Color.Firebrick;
						lastLabel.Text = w;
					}
				});
			}
			catch (Exception) { }   // the window may be gone
		});
	}

	void OnChildExit()
	{
		if (!running) return;
		int code = -1;
		try { code = child.ExitCode; } catch (Exception) { }
		// Whatever it started and left behind (a wsl.exe) goes with its job.
		if (childJob != null) { childJob.Kill(); childJob.Dispose(); childJob = null; }
		try { child.Dispose(); } catch (Exception) { }
		child = null;
		running = false;
		passBox.Text = "";   // typed again for the next run
		keyBox.Text = "";    // and so is the OpenAI key
		games.Clear();
		string end;
		bool clean = sawFinished && code == 0;
		if (stoppingNow) end = "Stopped now" + (wins + losses > 0 ? " after " + (wins + losses) + " games (" + wins + " - " + losses + ")" : "") +
			(abandoned > 0 ? ", " + abandoned + (abandoned == 1 ? " game" : " games") + " left to be lost on time" : "");
		else if (clean) end = (stoppingAfter ? "Stopped after the games in progress: " : "Finished: ") + wins + " - " + losses;
		else
		{
			string why = lastError ?? (lastStderr.Count > 0 ? lastStderr[lastStderr.Count - 1] : "exit code " + code);
			end = "The bot stopped: " + why;
			lastLabel.ForeColor = Color.Firebrick;
		}
		Log("---- the bot has ended (exit code " + code + ")");
		CloseRunLog();
		lastLabel.Text = end;
		if (clean || stoppingNow) lastLabel.ForeColor = Color.DimGray;
		UpdateButtons();
		ShowStatus();
		if (closeWhenDone) Close();
	}

	void CloseRunLog()
	{
		if (runLog != null) { try { runLog.Dispose(); } catch (Exception) { } runLog = null; }
	}

	void OnClosing(object sender, FormClosingEventArgs e)
	{
		if (!running) { tick.Stop(); return; }
		if (e.CloseReason == CloseReason.WindowsShutDown || e.CloseReason == CloseReason.TaskManagerClosing) return;   // the job ends the bot
		if (closeWhenDone)
		{
			// Already closing after the games in progress: offer to stop now instead.
			e.Cancel = true;
			if (!stoppingNow && MessageBox.Show(this, "The window closes when the games in progress are finished.\n\nStop now instead? They will be lost on time.",
				Program.AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
				StopNow();
			return;
		}
		e.Cancel = true;
		int choice;
		using (CloseDialog d = new CloseDialog(games.Count))
		{
			d.ShowDialog(this);
			choice = d.Choice;
		}
		if (choice == CloseDialog.Cancel) return;
		if (!running)
		{
			BeginInvoke((MethodInvoker)Close);   // it ended while the question was open: close as asked
			return;
		}
		closeWhenDone = true;
		if (choice == CloseDialog.Finish) StopAfter();
		else StopNow();
		UpdateButtons();
		ShowStatus();
	}

	// --------------------------------------------------------------------------------------
	// What the bot says
	// --------------------------------------------------------------------------------------

	void OnChildLine(string line, bool stderr)
	{
		// From a reader thread: written to the run's log here, shown on the window's thread.
		lock (outDone)
		{
			try { if (runLog != null) runLog.WriteLine((stderr ? "! " : "") + line); } catch (Exception) { }
		}
		try { BeginInvoke((MethodInvoker)delegate { HandleLine(line, stderr); }); } catch (Exception) { }
	}

	void HandleLine(string line, bool stderr)
	{
		if (!stderr && line.StartsWith("@app "))
		{
			Dictionary<string, object> ev = null;
			try { ev = json.DeserializeObject(line.Substring(5)) as Dictionary<string, object>; } catch (Exception) { }
			if (ev != null) { NoteThinking(ev); HandleEvent(ev); }
			return;
		}
		if (stderr)
		{
			if (line.Trim().Length == 0) return;
			lastStderr.Add(line.Trim());
			if (lastStderr.Count > 20) lastStderr.RemoveAt(0);
			Show("! " + line);
			return;
		}
		Show(line);
	}

	// A finished game's measured thinking (ladder.py's game_end: decision_ms and iterations, both means
	// over the game's decisions): one line in
	// the log, and the last game's under the search line.
	void NoteThinking(Dictionary<string, object> ev)
	{
		if (Str(ev, "ev") != "game_end") return;
		double ms, it;
		bool hasMs = double.TryParse(Str(ev, "decision_ms") ?? "", NumberStyles.Float, CultureInfo.InvariantCulture, out ms);
		bool hasIt = double.TryParse(Str(ev, "iterations") ?? "", NumberStyles.Float, CultureInfo.InvariantCulture, out it);
		if (!hasMs && !hasIt) return;
		lastThinking = "Last game: " +
			(hasMs ? "moves took " + (ms / 1000.0).ToString("0.0", CultureInfo.InvariantCulture) + " s on average" : "") +
			(hasMs && hasIt ? ", " : "") +
			(hasIt ? (it >= 1e6 ? (it / 1e6).ToString("0.0", CultureInfo.InvariantCulture) + "M" : (it / 1e3).ToString("0", CultureInfo.InvariantCulture) + "k") +
				" search iterations a decision on average" : "") + ".";
		lastThinkingLine = "Last game: " +
			(hasMs ? (ms / 1000.0).ToString("0.0", CultureInfo.InvariantCulture) + " s a move" : "") +
			(hasMs && hasIt ? ", " : "") +
			(hasIt ? (it >= 1e6 ? (it / 1e6).ToString("0.0", CultureInfo.InvariantCulture) + "M" : (it / 1e3).ToString("0", CultureInfo.InvariantCulture) + "k") +
				" iterations a decision" : "") + " on average.";
		Show("  " + lastThinking);
		UpdateSearchLabel();
	}

	static string Str(Dictionary<string, object> ev, string key)
	{
		object v;
		return ev.TryGetValue(key, out v) && v != null ? Convert.ToString(v, CultureInfo.InvariantCulture) : null;
	}

	static int? Int(Dictionary<string, object> ev, string key)
	{
		object v;
		if (!ev.TryGetValue(key, out v) || v == null) return null;
		try { return Convert.ToInt32(v, CultureInfo.InvariantCulture); } catch (Exception) { return null; }
	}

	static bool Bool(Dictionary<string, object> ev, string key)
	{
		object v;
		return ev.TryGetValue(key, out v) && v is bool && (bool)v;
	}

	void HandleEvent(Dictionary<string, object> ev)
	{
		string kind = Str(ev, "ev") ?? "";
		string tag = Str(ev, "battle");
		if (kind == "net") { NetEvent(ev, tag); return; }   // one a move: only the network panels change
		switch (kind)
		{
			case "phase":
				phase = Str(ev, "text") ?? "";
				break;
			case "start":
				sawStart = true;
				if (Int(ev, "earlier_rating").HasValue) ratingStart = ratingNow = Int(ev, "earlier_rating");
				if (ratingStart.HasValue && eloTrail.Count == 0) { eloTrail.Add(ratingStart.Value); eloView.Invalidate(); }
				Show("bot ready: " + Str(ev, "checkpoint") + " on " + Str(ev, "device") + ", " + Str(ev, "user") + " on " +
					(Str(ev, "server") == "public" ? "play.pokemonshowdown.com" : Str(ev, "server")));
				break;
			case "searching":
				break;
			case "game_start":
			{
				GameInfo g = new GameInfo();
				g.N = Int(ev, "n") ?? (gameNumbers.Count + 1);
				g.Opponent = Str(ev, "opponent");
				if (tag != null) { games[tag] = g; gameNumbers[tag] = g.N; }
				int? r = Int(ev, "rating");
				if (r.HasValue && !ratingStart.HasValue) ratingStart = r;
				if (r.HasValue && !ratingNow.HasValue) ratingNow = r;
				if (r.HasValue && eloTrail.Count == 0) { eloTrail.Add(r.Value); eloView.Invalidate(); }
				int? or = Int(ev, "opponent_rating");
				Show(string.Format("game {0} started vs {1}{2}", g.N, g.Opponent ?? "?", or.HasValue ? " (" + or.Value + ")" : ""));
				break;
			}
			case "move":
			{
				GameInfo g;
				int turn = Int(ev, "turn") ?? 0;
				if (tag != null && games.TryGetValue(tag, out g)) g.Turn = turn;
				int n;
				string who = tag != null && gameNumbers.TryGetValue(tag, out n) ? "game " + n : "game ?";
				Show(string.Format("  {0,-8} turn {1,2}: {2}{3}", who, turn, Choice(Str(ev, "choice")), Why(ev)));
				break;
			}
			case "game_end":
			{
				if (tag != null) games.Remove(tag);
				wins = Int(ev, "wins") ?? wins;
				losses = Int(ev, "losses") ?? losses;
				int? before = Int(ev, "rating_before"), after = Int(ev, "rating_after");
				if (!ratingStart.HasValue && before.HasValue) ratingStart = before;
				if (after.HasValue) ratingNow = after;
				if (after.HasValue)
				{
					if (eloTrail.Count == 0 && before.HasValue) eloTrail.Add(before.Value);
					if (eloTrail.Count > 0) { eloTrail.Add(after.Value); eloWon.Add(Bool(ev, "won")); }
					eloView.Invalidate();
				}
				if (netNow != null && tag != null && netNow.Tag == tag) { netNow.Ended = true; neuView.Invalidate(); }
				int n;
				lastLabel.ForeColor = Color.DimGray;
				lastLabel.Text = "Last game" + (tag != null && gameNumbers.TryGetValue(tag, out n) ? " (" + n + ")" : "") + ": " +
					(Bool(ev, "won") ? "WIN" : "LOSS") + " in " + Str(ev, "turns") + " turns vs " + (Str(ev, "opponent") ?? "?") +
					(before.HasValue && after.HasValue ? ", rating " + before + " -> " + after + " (" + ((after - before) >= 0 ? "+" : "") + (after - before) + ")" : "");
				break;
			}
			case "stopping":
				if (Str(ev, "how") == "now") { if (!stoppingNow) stopNowAt = DateTime.Now; stoppingNow = true; }
				else stoppingAfter = true;
				if (Str(ev, "how") == "health") lastError = "the search kept failing (" + Str(ev, "text") + ")";
				if (Str(ev, "how") == "model") lastError = Str(ev, "text");
				break;
			case "error":
				lastError = Str(ev, "text");
				break;
			case "finished":
				sawFinished = true;
				abandoned = Int(ev, "abandoned") ?? 0;
				break;
		}
		UpdateButtons();
		ShowStatus();
	}

	static string Choice(string c)
	{
		if (string.IsNullOrEmpty(c)) return "?";
		c = c.Trim();
		bool tera = c.EndsWith(" terastallize");
		if (tera) c = c.Substring(0, c.Length - " terastallize".Length);
		return c + (tera ? " + Tera" : "");
	}

	static string Why(Dictionary<string, object> ev)
	{
		if (!ev.ContainsKey("source")) return "";
		string by = Str(ev, "by");
		if (!string.IsNullOrEmpty(by))
		{
			// A model's game: whose move this is.
			string fell = Str(ev, "fell") ?? "";
			if (fell == "single") return "   (only move)";
			if (Bool(ev, "solo"))
				return fell.Length > 0 ? "   (the fallback rule: " + by + " gave no move, " + fell.Replace('_', ' ') + ")" : "   (" + by + "'s move)";
			if (fell.Length > 0) return "   (the bot's own move: " + by + " gave none, " + fell.Replace('_', ' ') + ")";
			return Bool(ev, "changed") ? "   (" + by + "'s pick, not the bot's own)" : "   (" + by + " agrees with the bot)";
		}
		if (Bool(ev, "lines_changed")) return "   (changed by reading ahead)";
		if (Bool(ev, "searched"))
		{
			int? it = Int(ev, "iterations");
			string its = it.HasValue ? (it.Value >= 1000 ? (it.Value / 1000) + "k" : it.Value.ToString()) + " iterations" : "searched";
			return "   (" + its + (Bool(ev, "overruled") ? ", the search overruled the network)" : ", search agrees)");
		}
		string reason = Str(ev, "reason") ?? "";
		if (reason == "single") return "   (only move)";
		if (reason == "revive") return "   (revival)";
		return "   (network" + (reason.Length > 0 ? ": search " + reason : "") + ")";
	}

	int shownLines;

	void Show(string text)
	{
		// The window keeps the last 3000 to 4000 lines (the run's log file keeps all of them).
		if (shownLines >= 4000)
		{
			string all = logBox.Text;
			int cut = -1;
			for (int i = 0; i < 1000; i++)
			{
				int next = all.IndexOf('\n', cut + 1);
				if (next < 0) break;
				cut = next;
			}
			if (cut > 0) { logBox.Text = all.Substring(cut + 1); shownLines -= 1000; }
		}
		logBox.AppendText(text + "\r\n");
		shownLines++;
	}

	void Log(string text)
	{
		Show(text);
		lock (outDone)
		{
			try { if (runLog != null) runLog.WriteLine("# " + text); } catch (Exception) { }
		}
	}

	void Warn(string text)
	{
		MessageBox.Show(this, text, Program.AppName, MessageBoxButtons.OK, MessageBoxIcon.Information);
	}

	// --------------------------------------------------------------------------------------
	// The panels on the right: this run's Elo, and the network at work
	// --------------------------------------------------------------------------------------

	static readonly Color EloLine = Color.FromArgb(120, 170, 255);
	static readonly Color WinDot = Color.FromArgb(80, 205, 125);
	static readonly Color LossDot = Color.FromArgb(240, 95, 85);
	static readonly Color Marker = Color.FromArgb(250, 210, 90);
	// Neurons: dark (silent) through red and orange to white (firing hardest).
	static readonly Color[] FireStops = { Color.FromArgb(14, 14, 20), Color.FromArgb(96, 18, 18), Color.FromArgb(205, 64, 22),
		Color.FromArgb(250, 165, 40), Color.FromArgb(255, 244, 180) };
	static readonly Color NeuronSilent = Color.FromArgb(30, 32, 42);   // a neuron that does not fire on this move

	ViewPanel AddView(string name, Control parent, Action<Graphics, Rectangle> painter)
	{
		ViewPanel p = new ViewPanel();
		p.Name = name;
		p.Dock = DockStyle.Fill;
		p.Painter = painter;
		parent.Controls.Add(p);
		return p;
	}

	static Color Ramp(double t, Color[] stops)
	{
		if (double.IsNaN(t) || t < 0) t = 0;
		if (t > 1) t = 1;
		double x = t * (stops.Length - 1);
		int i = Math.Min(stops.Length - 2, (int)x);
		double u = x - i;
		Color a = stops[i], b = stops[i + 1];
		return Color.FromArgb((int)(a.R + (b.R - a.R) * u), (int)(a.G + (b.G - a.G) * u), (int)(a.B + (b.B - a.B) * u));
	}

	// A JSON array as JavaScriptSerializer returns it (object[], or an ArrayList), else empty.
	static object[] AsArray(object v)
	{
		object[] a = v as object[];
		if (a != null) return a;
		System.Collections.ArrayList l = v as System.Collections.ArrayList;
		return l != null ? l.ToArray() : new object[0];
	}

	static object[] Arr(Dictionary<string, object> ev, string key)
	{
		object v;
		return ev.TryGetValue(key, out v) ? AsArray(v) : new object[0];
	}

	static double[][] Matrix(object v)
	{
		object[] rows = AsArray(v);
		double[][] m = new double[rows.Length][];
		for (int r = 0; r < rows.Length; r++)
		{
			object[] row = AsArray(rows[r]);
			m[r] = new double[row.Length];
			for (int c = 0; c < row.Length; c++) m[r][c] = Num(row[c]);
		}
		return m;
	}

	static double Num(object v)
	{
		try { return Convert.ToDouble(v, CultureInfo.InvariantCulture); } catch (Exception) { return 0.0; }
	}

	// "ironvaliant" -> "Ironvaliant": poke-env's species ids, readable enough in a few pixels.
	static string Pretty(string species)
	{
		if (string.IsNullOrEmpty(species)) return "";
		return char.ToUpperInvariant(species[0]) + species.Substring(1);
	}

	// ladder.py's "net" line (--app-net-view, net_viz.py): the move the network just made. One game
	// is followed at a time, so two games at once do not take turns on the panels: the first one
	// seen, until it ends, then the next one that moves.
	void NetEvent(Dictionary<string, object> ev, string tag)
	{
		if (tag == null) return;
		if (followTag != null && followTag != tag && games.ContainsKey(followTag)) return;
		NetFrame f = new NetFrame();
		f.Tag = tag;
		int n;
		f.Game = gameNumbers.TryGetValue(tag, out n) ? n : 0;
		f.Turn = Int(ev, "turn") ?? 0;
		f.Layers = Int(ev, "layers") ?? 0;
		object[] own = Arr(ev, "own"), foe = Arr(ev, "foe");
		f.Own = new string[own.Length];
		for (int i = 0; i < own.Length; i++) f.Own[i] = Convert.ToString(own[i], CultureInfo.InvariantCulture) ?? "";
		f.Foe = new string[foe.Length];
		for (int i = 0; i < foe.Length; i++) f.Foe[i] = Convert.ToString(foe[i], CultureInfo.InvariantCulture) ?? "";
		f.OwnActive = Int(ev, "own_active") ?? -1;
		f.FoeActive = Int(ev, "foe_active") ?? -1;
		object[] att = Arr(ev, "att");
		f.Att = new double[att.Length][];
		for (int r = 0; r < att.Length; r++)
		{
			object[] row = AsArray(att[r]);
			f.Att[r] = new double[row.Length];
			for (int c = 0; c < row.Length; c++) f.Att[r][c] = Num(row[c]);
		}
		try { f.Neu = Convert.FromBase64String(Str(ev, "neu") ?? ""); } catch (Exception) { f.Neu = new byte[0]; }
		object[] top = Arr(ev, "neu_max");
		f.NeuMax = new double[top.Length];
		for (int i = 0; i < top.Length; i++) f.NeuMax[i] = Num(top[i]);
		int rows = f.NeuMax.Length;
		if (rows == 0 || f.Att.Length != rows || f.Neu.Length == 0 || f.Neu.Length % rows != 0) return;   // not a line this window knows
		try
		{
			object[] top2 = Arr(ev, "top");
			if (top2.Length == rows)
			{
				f.Top = new int[rows][];
				for (int r = 0; r < rows; r++)
				{
					object[] t = AsArray(top2[r]);
					f.Top[r] = new int[t.Length];
					for (int i = 0; i < t.Length; i++) f.Top[r][i] = (int)Num(t[i]);
				}
				object[] edges = Arr(ev, "edges");
				f.Edges = new double[edges.Length][][];
				for (int r = 0; r < edges.Length; r++) f.Edges[r] = Matrix(edges[r]);
				object[] probs = Arr(ev, "probs");
				f.Probs = new double[probs.Length];
				for (int i = 0; i < probs.Length; i++) f.Probs[i] = Num(probs[i]);
				object[] acts = Arr(ev, "acts"), names = Arr(ev, "act_names");
				f.Acts = new int[acts.Length];
				f.ActNames = new string[acts.Length];
				for (int i = 0; i < acts.Length; i++)
				{
					f.Acts[i] = (int)Num(acts[i]);
					f.ActNames[i] = i < names.Length ? Convert.ToString(names[i], CultureInfo.InvariantCulture) : "action " + f.Acts[i];
				}
				f.OutEdges = Matrix(Arr(ev, "out_edges"));
				object[] ps = Arr(ev, "params");
				if (ps.Length == rows + 2)
				{
					f.Params = new double[ps.Length];
					for (int i = 0; i < ps.Length; i++) f.Params[i] = Math.Max(1.0, Num(ps[i]));
				}
				int k = f.Top[0].Length;
				bool fits = f.Edges.Length == rows - 1 && f.OutEdges.Length == f.Acts.Length;
				foreach (int[] t in f.Top) fits &= t.Length == k;
				foreach (double[][] m in f.Edges) { fits &= m.Length == k; foreach (double[] row in m) fits &= row.Length == k; }
				foreach (double[] row in f.OutEdges) fits &= row.Length == k;
				if (!fits) f.Top = null;      // drawn as the grid of every neuron instead
			}
		}
		catch (Exception) { f.Top = null; }
		netAt = DateTime.Now;
		if (animate) netTick.Start();
		foreach (double[] row in f.Att) if (row.Length != 14) return;
		followTag = tag;
		// Brightness compares across moves. Only a few percent of the neurons fire at all on a move and a
		// handful fire far harder than the rest, so the scale is the mean of each row's 8 strongest, falling
		// slowly (3% a move) and jumping up to a new higher one.
		int dff = f.Neu.Length / rows;
		if (neuScale == null || neuScale.Length != rows) neuScale = new double[rows];
		for (int i = 0; i < rows; i++)
		{
			byte[] row = new byte[dff];
			Array.Copy(f.Neu, i * dff, row, 0, dff);
			Array.Sort(row);
			double strong = 0;
			for (int j = dff - 8; j < dff; j++) strong += row[j] / 255.0 * f.NeuMax[i];
			neuScale[i] = Math.Max(strong / 8, neuScale[i] * 0.97);
		}
		netNow = f;
		neuView.Invalidate();
	}

	// The title line of a panel, with the game and turn on the right. Returns where the content starts.
	float ViewHeader(Graphics g, Rectangle r, string title, string right, float reserve = 0)
	{
		float k = g.DpiX / 96f, pad = 10 * k;
		using (Brush ink = new SolidBrush(ViewPanel.Ink), dim = new SolidBrush(ViewPanel.Dim))
		{
			g.DrawString(title, viewTitle, ink, pad, 7 * k);
			if (!string.IsNullOrEmpty(right))
			{
				SizeF s = g.MeasureString(right, viewSmall);
				g.DrawString(right, viewSmall, dim, r.Width - pad - reserve * k - s.Width, 7 * k + (viewTitle.GetHeight(g) - viewSmall.GetHeight(g)) / 2);
			}
		}
		return 7 * k + viewTitle.GetHeight(g) + 6 * k;
	}

	void ViewNote(Graphics g, Rectangle r, float top, string text)
	{
		float pad = 10 * g.DpiX / 96f;
		using (Brush dim = new SolidBrush(ViewPanel.Dim))
			g.DrawString(text, viewSmall, dim, new RectangleF(pad, top, r.Width - 2 * pad, r.Height - top - pad));
	}

	string FrameText(NetFrame f)
	{
		if (f == null) return "";
		return (f.Game > 0 ? "game " + f.Game + ", " : "") + "turn " + f.Turn + (f.Ended ? " (game over)" : "");
	}

	string WaitingText(string what)
	{
		if (!running) return "During a run: " + what;
		if (!runUsesNet) return "A model alone plays this run: there is no network to show.";
		return "Waiting for the network's first move.";
	}

	// This run's rating: before its first game, then after each game; a dot per game, green for a win.
	void PaintElo(Graphics g, Rectangle r)
	{
		float k = g.DpiX / 96f, pad = 10 * k;
		int w = 0;
		foreach (bool b in eloWon) if (b) w++;
		string right = eloWon.Count > 0 ? w + " won, " + (eloWon.Count - w) + " lost" : "";
		float top = ViewHeader(g, r, "Elo, this run", right);
		if (eloTrail.Count == 0)
		{
			ViewNote(g, r, top, running ? "Waiting for the first rating." : "Start a run: the rating after every game is drawn here.");
			return;
		}
		int first = eloTrail[0], last = eloTrail[eloTrail.Count - 1], d = last - first;
		using (Brush ink = new SolidBrush(ViewPanel.Ink), dim = new SolidBrush(ViewPanel.Dim),
			delta = new SolidBrush(d > 0 ? WinDot : d < 0 ? LossDot : ViewPanel.Dim))
		using (Font big = new Font("Segoe UI", 14F, FontStyle.Bold))
		{
			g.DrawString(last.ToString(CultureInfo.InvariantCulture), big, ink, pad, top - 2 * k);
			float x = pad + g.MeasureString(last.ToString(CultureInfo.InvariantCulture), big).Width;
			g.DrawString((d >= 0 ? "+" : "") + d, viewTitle, delta, x, top + (big.GetHeight(g) - viewTitle.GetHeight(g)) / 2);
			g.DrawString("from " + first, viewSmall, dim, x + g.MeasureString((d >= 0 ? "+" : "") + d + " ", viewTitle).Width,
				top + (big.GetHeight(g) - viewSmall.GetHeight(g)) / 2);
			top += big.GetHeight(g) + 2 * k;

			float labelW = g.MeasureString("88888", viewSmall).Width + 2 * k;
			float bottomLabels = viewSmall.GetHeight(g) + 2 * k;
			RectangleF p = new RectangleF(pad + labelW, top + 4 * k, r.Width - 2 * pad - labelW - 4 * k, r.Height - top - 4 * k - pad - bottomLabels);
			if (p.Width < 20 || p.Height < 20) return;
			int lo = int.MaxValue, hi = int.MinValue;
			foreach (int v in eloTrail) { lo = Math.Min(lo, v); hi = Math.Max(hi, v); }
			double span = Math.Max(30, hi - lo), mid = (hi + lo) / 2.0, yLo = mid - span * 0.6, yHi = mid + span * 0.6;
			Func<double, float> Y = v => p.Bottom - (float)((v - yLo) / (yHi - yLo)) * p.Height;
			Func<int, float> X = i => eloTrail.Count == 1 ? p.Left + p.Width / 2 : p.Left + p.Width * i / (eloTrail.Count - 1);
			using (Pen rule = new Pen(ViewPanel.Rule, k), start = new Pen(ViewPanel.Dim, k))
			{
				g.DrawLine(rule, p.Left, p.Bottom, p.Right, p.Bottom);
				start.DashStyle = DashStyle.Dash;
				g.DrawLine(start, p.Left, Y(first), p.Right, Y(first));
			}
			StringFormat rightAlign = new StringFormat();
			rightAlign.Alignment = StringAlignment.Far;
			rightAlign.FormatFlags = StringFormatFlags.NoWrap;
			float lh = viewSmall.GetHeight(g);
			g.DrawString(hi.ToString(CultureInfo.InvariantCulture), viewSmall, dim, new RectangleF(pad - 4 * k, Y(hi) - lh / 2, labelW - 2 * k, lh), rightAlign);
			if (lo != hi) g.DrawString(lo.ToString(CultureInfo.InvariantCulture), viewSmall, dim, new RectangleF(pad - 4 * k, Y(lo) - lh / 2, labelW - 2 * k, lh), rightAlign);
			// "start" at the right end of its dashed line, on the side away from where the line ended.
			float startY = last >= first ? Y(first) + 1 * k : Y(first) - lh - 1 * k;
			g.DrawString("start", viewSmall, dim, p.Right - g.MeasureString("start", viewSmall).Width, startY);
			g.DrawString("game 0", viewSmall, dim, p.Left - 2 * k, p.Bottom + 2 * k);
			string games = "game " + (eloTrail.Count - 1);
			g.DrawString(games, viewSmall, dim, p.Right - g.MeasureString(games, viewSmall).Width + 2 * k, p.Bottom + 2 * k);
			if (eloTrail.Count >= 2)
			{
				PointF[] pts = new PointF[eloTrail.Count];
				for (int i = 0; i < pts.Length; i++) pts[i] = new PointF(X(i), Y(eloTrail[i]));
				using (Pen line = new Pen(EloLine, 2 * k)) g.DrawLines(line, pts);
			}
			float rad = (eloTrail.Count > 40 ? 2f : 3.2f) * k;
			using (Pen ring = new Pen(ViewPanel.Dim, k)) g.DrawEllipse(ring, X(0) - rad, Y(first) - rad, 2 * rad, 2 * rad);
			for (int i = 1; i < eloTrail.Count; i++)
				using (Brush dot = new SolidBrush(i - 1 < eloWon.Count && eloWon[i - 1] ? WinDot : LossDot))
					g.FillEllipse(dot, X(i) - rad, Y(eloTrail[i]) - rad, 2 * rad, 2 * rad);
		}
	}

	// The decision token's feed-forward neurons (1024 a row) in each layer and the readout, one dot
	// each: dim grey when silent, red to white the harder it fires, against the row's strongest few
	// over recent moves (NetEvent), so a quiet move looks quiet.
	void PaintNeurons(Graphics g, Rectangle r)
	{
		NetFrame f = netNow;
		float top = ViewHeader(g, r, "Neurons firing", FrameText(f), 46);
		// The switch: a pill, filled when the 3D blocks are on.
		{
			float k = g.DpiX / 96f;
			SizeF ts = g.MeasureString("3D", viewSmall);
			togglePill = new RectangleF(r.Width - 10 * k - ts.Width - 14 * k, 7 * k, ts.Width + 14 * k, ts.Height + 3 * k);
			using (GraphicsPath pill = new GraphicsPath())
			{
				float d = togglePill.Height;
				pill.AddArc(togglePill.X, togglePill.Y, d, d, 90, 180);
				pill.AddArc(togglePill.Right - d, togglePill.Y, d, d, 270, 180);
				pill.CloseFigure();
				using (Brush fill = new SolidBrush(view3d ? Color.FromArgb(40, 150, 200) : ViewPanel.Back)) g.FillPath(fill, pill);
				using (Pen line = new Pen(view3d ? Color.FromArgb(120, 210, 240) : ViewPanel.Dim, 1f * k)) g.DrawPath(line, pill);
			}
			using (Brush t = new SolidBrush(view3d ? Color.White : ViewPanel.Dim))
				g.DrawString("3D", viewSmall, t, togglePill.X + 7 * k, togglePill.Y + 1.5f * k);
		}
		if (f == null)
		{
			ViewNote(g, r, top, WaitingText(view3d
				? "the network as 3D blocks, each as big as its parameters, every neuron a cell that lights up when it fires."
				: "the network at work for each move it makes: its strongest neurons layer by layer, lit as they fire, and the moves it leans to."));
			return;
		}
		if (view3d && f.Params != null) Paint3D(g, r, f, top);
		else if (f.Top != null) PaintDiagram(g, r, f, top);
		else PaintNeuronGrid(g, r, f, top);
	}

	static string Short(double n)
	{
		if (n >= 1e6) return (n / 1e6).ToString("0.0", CultureInfo.InvariantCulture) + "M";
		if (n >= 1e3) return (n / 1e3).ToString("0", CultureInfo.InvariantCulture) + "k";
		return n.ToString("0", CultureInfo.InvariantCulture);
	}

	static Color Shade(Color c, double f)
	{
		return Color.FromArgb(c.A, (int)Math.Max(0, Math.Min(255, c.R * f)), (int)Math.Max(0, Math.Min(255, c.G * f)), (int)Math.Max(0, Math.Min(255, c.B * f)));
	}

	// Silent cells dark blue, firing ones through cyan to white.
	static readonly Color[] CellStops = { Color.FromArgb(18, 24, 40), Color.FromArgb(24, 70, 150), Color.FromArgb(40, 170, 220),
		Color.FromArgb(150, 235, 245), Color.FromArgb(255, 255, 255) };

	// The network as 3D blocks, the way architecture diagrams draw it: the input encoder, the 3
	// transformer layers, the readout and the move head, each block's face as big as its parameters
	// (square root), so the encoder is the largest and the head the smallest. On each face every
	// neuron is a cell: the layers' and readout's 1024 feed-forward neurons of the decision token,
	// the encoder's 14 inputs by the readout's attention (field, 12 Pokemon, memory), the head's 14
	// actions by probability. After each move the blocks light up left to right, each cell flashing
	// as it fires. No lines between them: they distracted (2026-10-15).
	void Paint3D(Graphics g, Rectangle r, NetFrame f, float top)
	{
		float k = g.DpiX / 96f, pad = 10 * k;
		int rows = f.NeuMax.Length, dff = f.Neu.Length / rows, blocks = rows + 2;
		float lh = viewSmall.GetHeight(g);
		float areaTop = top + 2 * k, areaBottom = r.Height - pad - (lh + 4 * k) - (2 * lh + 6 * k);
		float areaH = areaBottom - areaTop, areaW = r.Width - 2 * pad;
		if (areaH < 40 || areaW < 60) { PaintNeuronGrid(g, r, f, top); return; }
		double maxP = 0;
		foreach (double p in f.Params) maxP = Math.Max(maxP, p);
		float[] h = new float[blocks], w = new float[blocks], dx = new float[blocks], dy = new float[blocks];
		float gap = 5 * k, need = 0, tall = 0;
		for (int b = 0; b < blocks; b++)
		{
			h[b] = (float)Math.Max(0.14, Math.Sqrt(f.Params[b] / maxP));
			w[b] = h[b] * 0.22f;
			dx[b] = h[b] * 0.20f;
			dy[b] = h[b] * 0.14f;
			need += w[b] + dx[b];
			tall = Math.Max(tall, h[b] + dy[b]);
		}
		float scale = Math.Min((areaW - gap * (blocks - 1)) / need, areaH / tall);
		for (int b = 0; b < blocks; b++) { h[b] *= scale; w[b] *= scale; dx[b] *= scale; dy[b] *= scale; }
		float used = gap * (blocks - 1);
		for (int b = 0; b < blocks; b++) used += w[b] + dx[b];
		float[] x = new float[blocks], y = new float[blocks];
		float cx = pad + (areaW - used) / 2, mid = areaTop + areaH / 2;
		for (int b = 0; b < blocks; b++)
		{
			x[b] = cx;
			y[b] = mid - (h[b] - dy[b]) / 2;      // the block with its top face, centred
			cx += w[b] + dx[b] + gap;
		}

		double ms = animate ? (DateTime.Now - netAt).TotalMilliseconds : 1e9;
		double step = (SweepMs - 320) / blocks;
		Func<int, double> Lit = b => Clamp01((ms - b * step) / 200.0);
		Func<int, double> Pulse = b => animate ? Math.Sin(Math.PI * Clamp01((ms - b * step) / 340.0)) : 0.0;

		// Each block's cells: count, grid and value (0..1) of each.
		Func<int, int> Cells = b => b == 0 || b == blocks - 1 ? 14 : dff;
		Func<int, int> Cols = b => b == 0 ? 2 : b == blocks - 1 ? 1 : 16;
		double[] attTop = f.Att[rows - 1];
		double attMax = 1e-9, probMax = 1e-9;
		foreach (double v in attTop) attMax = Math.Max(attMax, v);
		foreach (double v in f.Probs) probMax = Math.Max(probMax, v);
		Func<int, int, double> Value = (b, i) =>
			b == 0 ? attTop[i] / attMax
			: b == blocks - 1 ? (i < f.Probs.Length ? f.Probs[i] / probMax : 0.0)
			: f.Neu[(b - 1) * dff + i] / 255.0;
		Func<int, int, PointF> CellCenter = (b, i) =>
		{
			int cols = Cols(b), lines = (Cells(b) + cols - 1) / cols;
			return new PointF(x[b] + w[b] * ((i % cols) + 0.5f) / cols, y[b] + h[b] * ((i / cols) + 0.5f) / lines);
		};
		Color[] tint = new Color[blocks];
		for (int b = 0; b < blocks; b++)
			tint[b] = b == 0 ? Color.FromArgb(40, 120, 130) : b == blocks - 1 ? Color.FromArgb(150, 120, 40) : b == blocks - 2 ? Color.FromArgb(95, 70, 150) : Color.FromArgb(45, 80, 150);

		for (int b = 0; b < blocks; b++)
		{
			double lit = Lit(b), pulse = Pulse(b);
			int n = Cells(b), cols = Cols(b), lines = (n + cols - 1) / cols;
			double mean = 0;
			for (int i = 0; i < n; i++) mean += Value(b, i);
			mean /= n;
			double glow = 0.55 + 0.9 * mean * lit + 0.6 * pulse * Math.Sqrt(mean);
			PointF[] topFace = { new PointF(x[b], y[b]), new PointF(x[b] + w[b], y[b]), new PointF(x[b] + w[b] + dx[b], y[b] - dy[b]), new PointF(x[b] + dx[b], y[b] - dy[b]) };
			PointF[] side = { new PointF(x[b] + w[b], y[b]), new PointF(x[b] + w[b] + dx[b], y[b] - dy[b]), new PointF(x[b] + w[b] + dx[b], y[b] + h[b] - dy[b]), new PointF(x[b] + w[b], y[b] + h[b]) };
			using (Brush tb = new SolidBrush(Shade(tint[b], 1.25 * glow))) g.FillPolygon(tb, topFace);
			using (Brush sb = new SolidBrush(Shade(tint[b], 0.75 * glow))) g.FillPolygon(sb, side);
			// The face: one cell per neuron.
			using (Bitmap bmp = new Bitmap(cols, lines, PixelFormat.Format32bppArgb))
			{
				for (int i = 0; i < cols * lines; i++)
				{
					if (i >= n) { bmp.SetPixel(i % cols, i / cols, CellStops[0]); continue; }
					double v = Value(b, i);
					double shown = Math.Min(1.0, v * lit + 0.7 * pulse * v);   // a flash as the cell fires, then its level
					bmp.SetPixel(i % cols, i / cols, v <= 0 ? CellStops[0] : Ramp(0.2 + 0.8 * Math.Sqrt(shown), CellStops));
				}
				InterpolationMode im = g.InterpolationMode;
				PixelOffsetMode pm = g.PixelOffsetMode;
				g.InterpolationMode = InterpolationMode.NearestNeighbor;
				g.PixelOffsetMode = PixelOffsetMode.Half;
				g.DrawImage(bmp, new RectangleF(x[b], y[b], w[b], h[b]));
				g.InterpolationMode = im;
				g.PixelOffsetMode = pm;
			}
			if (n <= 14)       // few cells: a line around each, so they read as neurons
				using (Pen grid = new Pen(Color.FromArgb(120, 10, 12, 18), 1f * k))
				{
					for (int c = 1; c < cols; c++) g.DrawLine(grid, x[b] + w[b] * c / cols, y[b], x[b] + w[b] * c / cols, y[b] + h[b]);
					for (int l = 1; l < lines; l++) g.DrawLine(grid, x[b], y[b] + h[b] * l / lines, x[b] + w[b], y[b] + h[b] * l / lines);
				}
			using (Pen edge = new Pen(Shade(tint[b], 1.7 * glow), 1f * k))
			{
				g.DrawRectangle(edge, x[b], y[b], w[b], h[b]);
				g.DrawPolygon(edge, topFace);
				g.DrawPolygon(edge, side);
			}
		}

		// Under each block its name and size; under the layers the share of neurons that fire.
		using (Brush ink = new SolidBrush(ViewPanel.Ink), dim = new SolidBrush(ViewPanel.Dim), mark = new SolidBrush(Marker))
		{
			StringFormat center = new StringFormat();
			center.Alignment = StringAlignment.Center;
			center.FormatFlags = StringFormatFlags.NoWrap;
			for (int b = 0; b < blocks; b++)
			{
				string name = b == 0 ? "In" : b == blocks - 1 ? "Move" : b == blocks - 2 ? "Out" : "L" + b;
				float bx = x[b] + (w[b] + dx[b]) / 2;
				RectangleF cell = new RectangleF(bx - 30 * k, areaBottom + 3 * k, 60 * k, lh);
				g.DrawString(name, viewSmall, b == blocks - 1 ? mark : ink, cell, center);
				cell.Y += lh;
				string sub = Short(f.Params[b]);
				if (b > 0 && b < blocks - 1)
				{
					int firing = 0;
					for (int j = 0; j < dff; j++) if (f.Neu[(b - 1) * dff + j] > 0) firing++;
					sub += " " + (100.0 * firing / dff).ToString("0", CultureInfo.InvariantCulture) + "%";
				}
				g.DrawString(sub, viewSmall, dim, cell, center);
			}
			string plays = f.Acts != null && f.Acts.Length > 0 && f.ActNames != null
				? "It leans to " + PrettyAction(f.ActNames[0]) + " (" + (100 * f.Probs[f.Acts[0]]).ToString("0", CultureInfo.InvariantCulture) + "%)."
				: "";
			g.DrawString("Sized by parameters; a cell per neuron. " + plays, viewSmall, dim,
				new RectangleF(pad, r.Height - pad - lh, r.Width - 2 * pad, lh + 2 * k));
		}
	}

	static double Clamp01(double v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

	// "earthpower + Tera" -> "Earthpower + Tera", "switch garchomp" -> "Switch Garchomp".
	static string PrettyAction(string a)
	{
		if (string.IsNullOrEmpty(a)) return "?";
		string[] words = a.Split(' ');
		for (int i = 0; i < words.Length; i++) words[i] = Pretty(words[i]);
		return string.Join(" ", words);
	}

	// In the style of 3Blue1Brown's networks: each row (the 3 layers and the readout) a column of its
	// 10 strongest neurons, white as they fire, then the moves it leans to, as bars. No lines between
	// them (they distracted, 2026-10-15). After each move the rows light up left to right (SweepMs).
	void PaintDiagram(Graphics g, Rectangle r, NetFrame f, float top)
	{
		float k = g.DpiX / 96f, pad = 10 * k;
		int rows = f.Top.Length, n = f.Top[0].Length, dff = f.Neu.Length / rows, acts = f.Acts.Length;
		float lh = viewSmall.GetHeight(g);
		float areaTop = top + 2 * k;
		float areaBottom = r.Height - pad - (lh + 6 * k) - (2 * lh + 4 * k);
		float outW = Math.Min(r.Width * 0.36f, 118 * k);
		float outLeft = r.Width - pad - outW;
		float netLeft = pad + 10 * k, netRight = outLeft - 18 * k;
		if (areaBottom - areaTop < 40 || netRight - netLeft < 40) { PaintNeuronGrid(g, r, f, top); return; }
		float spacing = (areaBottom - areaTop) / n;
		float rad = Math.Min(spacing * 0.36f, 7.5f * k);
		Func<int, float> X = c => netLeft + rad + (netRight - netLeft - 2 * rad) * c / Math.Max(1, rows - 1);
		Func<int, float> Y = i => areaTop + spacing * (i + 0.5f);
		float outRow = (areaBottom - areaTop) / Math.Max(1, TopActionsShown);
		Func<int, float> OY = a => areaTop + outRow * (a + 0.5f);
		double ms = animate ? (DateTime.Now - netAt).TotalMilliseconds : 1e9;
		double step = (SweepMs - 260) / rows;
		Func<int, double> Lit = c => Clamp01((ms - c * step) / 220.0);              // row c lighting up
		Func<int, int, double> Act = (row, i) => f.Neu[row * dff + f.Top[row][i]] / 255.0;
		Color pos = Color.FromArgb(88, 196, 221), neg = Color.FromArgb(252, 98, 85);

		// The neurons: dark when silent, white as they fire, a soft glow on the strongest.
		using (Pen ring = new Pen(Color.FromArgb(150, 156, 168), 1f * k))
			for (int c = 0; c < rows; c++)
			{
				double lit = Lit(c);
				for (int i = 0; i < n; i++)
				{
					double a = Act(c, i) * lit;
					float x = X(c), y = Y(i);
					if (a > 0.55)
						using (Brush glow = new SolidBrush(Color.FromArgb((int)(90 * (a - 0.55) / 0.45), 255, 244, 214)))
							g.FillEllipse(glow, x - rad * 1.9f, y - rad * 1.9f, rad * 3.8f, rad * 3.8f);
					int v = (int)(18 + 237 * a);
					using (Brush fill = new SolidBrush(Color.FromArgb(v, v, Math.Min(255, v + 6))))
						g.FillEllipse(fill, x - rad, y - rad, 2 * rad, 2 * rad);
					g.DrawEllipse(ring, x - rad, y - rad, 2 * rad, 2 * rad);
				}
			}

		using (Brush ink = new SolidBrush(ViewPanel.Ink), dim = new SolidBrush(ViewPanel.Dim))
		{
			// Each row's name, and the share of all its neurons that fire on this move.
			StringFormat center = new StringFormat();
			center.Alignment = StringAlignment.Center;
			center.FormatFlags = StringFormatFlags.NoWrap;
			for (int c = 0; c < rows; c++)
			{
				int firing = 0;
				for (int j = 0; j < dff; j++) if (f.Neu[c * dff + j] > 0) firing++;
				RectangleF cell = new RectangleF(X(c) - 26 * k, areaBottom + 3 * k, 52 * k, lh);
				g.DrawString(c < f.Layers ? "L" + (c + 1) : "Out", viewSmall, c < f.Layers ? dim : ink, cell, center);
				cell.Y += lh;
				g.DrawString((100.0 * firing / dff).ToString("0", CultureInfo.InvariantCulture) + "%", viewSmall, dim, cell, center);
			}
			// The moves it leans to: name, bar, probability. The likeliest in yellow.
			double grow = Lit(rows);
			float barMax = outW - 4 * k;
			for (int a = 0; a < Math.Min(acts, TopActionsShown); a++)
			{
				double p = f.Acts[a] < f.Probs.Length ? f.Probs[f.Acts[a]] : 0.0;
				float y = OY(a) - (lh + 7 * k) / 2;
				Color c = a == 0 ? Marker : pos;
				using (Brush name = new SolidBrush(a == 0 ? Marker : ViewPanel.Ink))
				{
					StringFormat clip = new StringFormat(StringFormatFlags.NoWrap);
					clip.Trimming = StringTrimming.EllipsisCharacter;
					g.DrawString(PrettyAction(f.ActNames[a]), viewSmall, name, new RectangleF(outLeft, y, outW, lh), clip);
				}
				float bw = (float)(barMax * p * grow);
				using (Brush bar = new SolidBrush(Color.FromArgb(a == 0 ? 230 : 170, c)))
					g.FillRectangle(bar, outLeft, y + lh + 1 * k, Math.Max(1f * k, bw), 5 * k);
				string pct = (100 * p).ToString(p >= 0.095 ? "0" : "0.0", CultureInfo.InvariantCulture) + "%";
				g.DrawString(pct, viewSmall, dim, outLeft + Math.Max(1f * k, bw) + 3 * k, y + lh - 3 * k);
			}
			g.DrawString("its move", viewSmall, dim, new RectangleF(outLeft, areaBottom + 3 * k, outW, lh));
			g.DrawString("The " + n + " strongest neurons of each layer, lit as they fire.", viewSmall, dim,
				new RectangleF(pad, r.Height - pad - lh, r.Width - 2 * pad, lh + 2 * k));
		}
	}

	const int TopActionsShown = 4;

	// The older picture, for a line without the diagram: every neuron of each row as a dot.
	void PaintNeuronGrid(Graphics g, Rectangle r, NetFrame f, float top)
	{
		float k = g.DpiX / 96f, pad = 10 * k;
		int rows = f.NeuMax.Length, dff = f.Neu.Length / rows;
		float lh = viewSmall.GetHeight(g);
		float labelW = g.MeasureString("Out ", viewSmall).Width + 2 * k;
		float litW = g.MeasureString("100% firing", viewSmall).Width + 4 * k;
		float areaW = r.Width - 2 * pad - labelW - litW;
		float legendH = lh + 6 * k;
		float rowGap = 6 * k;
		float stripH = (r.Height - top - pad - legendH - rowGap * (rows - 1)) / rows;
		if (areaW < 16 || stripH < 4) return;
		// Columns so the dots are as large as they can be and square.
		int cols = 64;
		float size = 0;
		foreach (int c in new int[] { 256, 128, 64, 32, 16 })
		{
			if (dff % c != 0) continue;
			float s = Math.Min(areaW / c, stripH / (dff / c));
			if (s > size) { size = s; cols = c; }
		}
		int lines = dff / cols;
		g.InterpolationMode = InterpolationMode.NearestNeighbor;
		g.PixelOffsetMode = PixelOffsetMode.Half;
		using (Brush ink = new SolidBrush(ViewPanel.Ink), dim = new SolidBrush(ViewPanel.Dim))
		{
			for (int row = 0; row < rows; row++)
			{
				float y = top + row * (stripH + rowGap) + (stripH - size * lines) / 2;
				double scale = neuScale != null && row < neuScale.Length && neuScale[row] > 1e-6 ? neuScale[row] : Math.Max(f.NeuMax[row], 1e-6);
				int firing = 0;
				using (Bitmap bmp = new Bitmap(cols, lines, PixelFormat.Format32bppArgb))
				{
					for (int j = 0; j < dff; j++)
					{
						byte q = f.Neu[row * dff + j];
						if (q == 0) { bmp.SetPixel(j % cols, j / cols, NeuronSilent); continue; }
						firing++;
						// Every neuron that fires shows; the square root spreads the weak ones out from the strong.
						double a = Math.Min(1.0, q / 255.0 * f.NeuMax[row] / scale);
						bmp.SetPixel(j % cols, j / cols, Ramp(0.18 + 0.82 * Math.Sqrt(a), FireStops));
					}
					g.DrawImage(bmp, new RectangleF(pad + labelW, y, size * cols, size * lines));
				}
				g.DrawString(row < f.Layers ? "L" + (row + 1) : "Out", viewSmall, row < f.Layers ? dim : ink, pad, y + (size * lines - lh) / 2);
				string pct = (100.0 * firing / dff).ToString("0", CultureInfo.InvariantCulture) + "% firing";
				g.DrawString(pct, viewSmall, dim, pad + labelW + size * cols + 4 * k, y + (size * lines - lh) / 2);
			}
			g.DrawString(dff + " neurons a layer. Brighter fires harder.", viewSmall, dim,
				new RectangleF(pad, r.Height - pad - lh, r.Width - 2 * pad, lh + 2 * k));
		}
	}

	// --test-replay: the "@app" lines of a recorded run, handled as if a run printed them now (no run
	// is started, nothing is sent anywhere). With --test-snapshot the window is saved as a PNG and closes.
	void Replay()
	{
		animate = false;
		try
		{
			foreach (string line in File.ReadAllLines(opts.ReplayFile, Encoding.UTF8))
				if (line.StartsWith("@app ")) HandleLine(line, false);
		}
		catch (Exception e) { Log("could not replay " + opts.ReplayFile + ": " + e.Message); }
		if (string.IsNullOrEmpty(opts.SnapshotFile)) return;
		BeginInvoke((MethodInvoker)delegate
		{
			try
			{
				Refresh();
				using (Bitmap b = new Bitmap(Width, Height))
				{
					DrawToBitmap(b, new Rectangle(0, 0, Width, Height));
					b.Save(opts.SnapshotFile, ImageFormat.Png);
				}
			}
			catch (Exception e) { Log("could not save " + opts.SnapshotFile + ": " + e.Message); }
			Close();
		});
	}

	// --------------------------------------------------------------------------------------
	// The Elo graph and the results
	// --------------------------------------------------------------------------------------

	string ResultsFile()
	{
		return opts.Test ? System.IO.Path.Combine(appDir, "runs", "ladder-test", "results.jsonl")
			: System.IO.Path.Combine(appDir, "runs", "ladder", "results.jsonl");
	}

	void ShowGraph()
	{
		if (graphRunning) return;
		graphRunning = true;
		UpdateButtons();
		List<string> args = new List<string> { "-u", "ladder.py", "--graph", "--once", "--out", ResultsFile(),
			"--server", opts.Test ? "local:" + opts.TestPort : "public" };
		string user = userBox.Text.Trim();
		if (user.Length > 0) { args.Add("--user"); args.Add(user); }
		Show("Elo graph: writing the page ...");
		Thread t = new Thread(delegate ()
		{
			string output;
			try { output = RunPythonQuick(args.ToArray(), null, 120000); }
			catch (Exception e) { output = "the graph could not be made: " + e.Message; }
			if (IsDisposed) return;
			try { BeginInvoke((MethodInvoker)delegate
			{
				graphRunning = false;
				UpdateButtons();
				string path = null;
				foreach (string l in output.Split('\n'))
				{
					string s = l.Trim();
					if (s.StartsWith("graph: ")) path = s.Substring(7).Trim();
					else if (s.Length > 0 && !s.StartsWith("@app ")) Show("Elo graph: " + s);
				}
				if (path != null && File.Exists(path))
				{
					Show("Elo graph: " + path);
					try { Process.Start(new ProcessStartInfo(path) { UseShellExecute = true }); }
					catch (Exception e) { Show("Elo graph: could not open it: " + e.Message); }
				}
			}); } catch (Exception) { }   // the window may be gone
		});
		t.IsBackground = true;
		t.Start();
	}

	void OpenFolder()
	{
		string dir = System.IO.Path.GetDirectoryName(ResultsFile());
		try
		{
			Directory.CreateDirectory(dir);
			Process.Start(new ProcessStartInfo(dir) { UseShellExecute = true });
		}
		catch (Exception e) { Show("could not open " + dir + ": " + e.Message); }
	}
}
