import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";
import { Cpu } from "lucide-react";

export const metadata: Metadata = {
  title: "Scalar - Autoscaling GPU Compute",
  description: "Deploy and manage autoscaling GPU compute workloads",
};

function Navigation() {
  return (
    <nav className="border-b border-gray-300 bg-white">
      <div className="max-w-7xl mx-auto px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 font-semibold text-xl">
              <Cpu className="w-6 h-6 text-indigo-600" />
              <span className="font-serif">Scalar</span>
            </Link>
            <div className="flex items-center gap-6">
              <Link
                href="/"
                className="text-gray-700 hover:text-gray-900 transition-colors"
              >
                Home
              </Link>
              <Link
                href="/apps"
                className="text-gray-700 hover:text-gray-900 transition-colors"
              >
                Apps
              </Link>
              <Link
                href="/deploy"
                className="text-gray-700 hover:text-gray-900 transition-colors"
              >
                Deploy
              </Link>
              <Link
                href="/resources"
                className="text-gray-700 hover:text-gray-900 transition-colors"
              >
                Resources
              </Link>
              <Link
                href="/orderbook"
                className="text-gray-700 hover:text-gray-900 transition-colors"
              >
                Order Book
              </Link>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/apps"
              className="text-sm text-gray-600 hover:text-gray-900"
            >
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-white">
        <Navigation />
        <main>{children}</main>
      </body>
    </html>
  );
}
