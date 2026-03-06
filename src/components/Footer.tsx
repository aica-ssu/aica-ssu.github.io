export default function Footer() {
  return (
    <footer className="border-t py-8 mt-16" style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-secondary)" }}>
      <div className="max-w-6xl mx-auto px-4 text-center text-sm" style={{ color: "var(--text-muted)" }}>
        <p>&copy; {new Date().getFullYear()} AICA Lab, Soongsil University. All rights reserved.</p>
        <p className="mt-1">Information Science Building, Room 306 &middot; Seoul, South Korea</p>
      </div>
    </footer>
  );
}
