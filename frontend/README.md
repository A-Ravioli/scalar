# Scalar Frontend

A modern Next.js frontend for managing autoscaling GPU compute workloads.

## Features

- 🚀 Deploy GPU compute applications with autoscaling
- 📊 Monitor running applications in real-time
- 💻 Clean, minimalist UI inspired by SF Compute
- 🔄 Auto-refreshing data for live updates
- 📱 Responsive design for mobile and desktop

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe code
- **Tailwind CSS v4** - Utility-first styling
- **Lucide React** - Beautiful icons
- **Recharts** - Data visualization (ready for future charts)

## Getting Started

### Prerequisites

- Node.js 18+ installed
- The Scalar backend API running (default: `http://localhost:8000`)

### Installation

1. Install dependencies:

```bash
npm install
```

2. Set up environment variables:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` with your configuration:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=sk_your_api_key
```

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### Production Build

Build for production:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with navigation
│   ├── page.tsx           # Home page
│   ├── deploy/            # Deploy new applications
│   ├── apps/              # List and manage applications
│   │   └── [id]/         # Individual app details
│   └── resources/         # View capacity
├── components/            # Reusable React components
│   ├── AppCard.tsx       # Application card display
│   ├── StatusBadge.tsx   # Status indicator
│   └── TierSelector.tsx  # Tier selection UI
├── lib/                   # Utilities and API client
│   ├── api.ts            # API client
│   └── types.ts          # TypeScript types
└── public/               # Static assets
```

## Pages

### Home (`/`)
- Overview dashboard with stats
- Recent applications list
- Quick access to deploy

### Deploy (`/deploy`)
- Create new GPU compute applications
- Configure resources (GPUs, CPU, RAM)
- Select tier (FAST or FLEX)
- Set Docker image and environment

### Apps (`/apps`)
- List all applications
- Filter by status (All, Running, Pending, Completed)
- Auto-refresh every 5 seconds
- Delete applications

### App Detail (`/apps/[id]`)
- Detailed application information
- Configuration details
- Allocation information (node, GPUs)
- Runtime metrics
- Delete action

### Resources (`/resources`)
- View cluster capacity
- GPU availability breakdown
- Node statistics
- Auto-refresh every 10 seconds

## Design System

The UI follows a clean, minimalist aesthetic:

- **Typography**: Merriweather serif for headings, Inter sans-serif for body
- **Colors**: White backgrounds, gray borders (#E5E7EB), indigo accents (#4F46E5)
- **Layout**: Generous padding, max-width containers, card-based design
- **Style**: Subtle borders, minimal shadows, clean lines

## API Integration

The frontend connects to the Scalar backend API:

- `POST /jobs` - Create application
- `GET /jobs` - List applications
- `GET /jobs/{id}` - Get application details
- `POST /jobs/{id}/cancel` - Delete application
- `GET /capacity_snapshot` - Get capacity information

## Development Notes

- Auto-refresh is enabled on list pages to show real-time updates
- The backend uses "jobs" terminology, but the frontend presents them as "apps"
- Error handling is built-in for all API calls
- Loading states are shown during async operations

## Future Enhancements

- View application logs
- Real-time metrics and charts
- WebSocket integration for live updates
- User authentication and multi-tenancy
- Cost tracking and billing integration
- Advanced filtering and search

## License

Private - Part of the Scalar project
