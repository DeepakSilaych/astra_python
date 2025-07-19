# purecv Reflection Microservice

A **FastAPI**-based microservice for generating image reflections using the `purecv` library.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [API Endpoints](#api-endpoints)
  - [Query Parameters](#query-parameters)
  - [Example Requests](#example-requests)
- [Development](#development)
- [Dockerization](#dockerization)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

- Removes the background of an uploaded image and generates a reflection.
- Reflection opacity, blur radius, vertical offset, and fade curve are fully configurable.
- Keeps the original background intact.
- Returns the processed image via HTTP response.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/python-microservices.git
   cd python-microservices
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

### API Endpoints

#### GET /

Returns a simple welcome message.

- **Request:**
  ```bash
  curl http://127.0.0.1:8000/
  ```
- **Response:**
  ```json
  { "message": "purecv reflection microservice" }
  ```

#### POST /reflect

Generates a reflection for an image, supporting multiple input methods:

- **Multipart Form-Data**:
  - `file`: upload an image file.
  - `url`: URL of a remote image to fetch.
  - `base64_str`: base64-encoded image string or data URI.
- **JSON Payload** (`Content-Type: application/json`):
  - `{ "url": "<image_url>" }`
  - `{ "base64": "<base64_data_or_data_uri>" }`

**Query Parameters** (see below table):

- `reflection_strength`, `blur_radius`, `reflection_y_offset`, `fade_power`, and `sync`.

**Behavior**:

- If `sync=true`, the call blocks until processing completes and returns the reflected image directly (`image/png`).
- Otherwise, the call immediately returns a JSON object `{ "task_id": "<id>" }` which can be polled.

**Query Parameters**
| Name | Type | Default | Description |
|---------------------|---------|---------|-----------------------------------------------------------------------------------------------|
| reflection_strength | float | 0.4 | Opacity of the reflection's top (0.0 to 1.0) |
| blur_radius | float | 1.0 | Gaussian blur radius applied to the reflection |
| reflection_y_offset | int | 0 | Vertical gap (in pixels) between the object and its reflection |
| fade_power | float | 1.5 | Controls the fade-off curve; values >1 fade faster initially |
| sync | boolean | false | If `true`, returns the processed image directly; otherwise returns a task ID to poll |

### Example Requests

#### 1) File upload (synchronous)

```bash
curl -X POST "http://127.0.0.1:8000/reflect?sync=true" \
  -F "file=@/path/to/image.png" \
  --output reflected.png
```

#### 2) File upload (asynchronous)

```bash
curl -X POST "http://127.0.0.1:8000/reflect" \
  -F "file=@/path/to/image.png"
```

#### 3) URL input (JSON, synchronous)

```bash
curl -X POST "http://127.0.0.1:8000/reflect?sync=true" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  --output reflected.png
```

#### 4) Base64 input (JSON, asynchronous)

```bash
curl -X POST "http://127.0.0.1:8000/reflect" \
  -H "Content-Type: application/json" \
  -d '{"base64":"data:image/png;base64,AAA..."}'
```

### JavaScript/TypeScript Examples

// Synchronous URL example using node-fetch and fs

```javascript
import fetch from "node-fetch";
import fs from "fs";

async function reflectFromUrl() {
  const res = await fetch("http://127.0.0.1:8000/reflect?sync=true", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: "https://example.com/image.jpg" }),
  });
  if (!res.ok) throw new Error(`Error: ${res.status}`);
  const arrayBuffer = await res.arrayBuffer();
  fs.writeFileSync("reflected.png", Buffer.from(arrayBuffer));
}
reflectFromUrl();
```

// Asynchronous file upload with polling using axios

```typescript
import axios from "axios";
import FormData from "form-data";
import fs from "fs";

async function reflectFileAsync() {
  const form = new FormData();
  form.append("file", fs.createReadStream("back_view_1747749326075.png"));

  const enqueue = await axios.post("http://127.0.0.1:8000/reflect", form, {
    headers: form.getHeaders(),
  });
  const { task_id } = enqueue.data;
  console.log("Task ID:", task_id);

  // Poll until we get an image
  while (true) {
    await new Promise((r) => setTimeout(r, 1000));
    const poll = await axios.get(`http://127.0.0.1:8000/reflect/${task_id}`, {
      responseType: "arraybuffer",
      validateStatus: (status) => status < 500,
    });
    if (poll.headers["content-type"].startsWith("image/")) {
      fs.writeFileSync("reflected.png", Buffer.from(poll.data));
      console.log("Reflection saved to reflected.png");
      break;
    } else {
      console.log(
        "Status:",
        JSON.parse(Buffer.from(poll.data).toString()).status,
      );
    }
  }
}
reflectFileAsync();
```

### Next.js Full-Stack Integration

#### Enabling CORS in FastAPI

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your Next.js origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Direct Client Calls from Next.js

In a Next.js page (e.g. `pages/index.tsx`):

```tsx
import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/reflect?sync=true", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      console.error(await res.text());
      return;
    }
    const blob = await res.blob();
    setImageUrl(URL.createObjectURL(blob));
  };

  return (
    <div>
      <h1>Reflect an Image</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button onClick={handleUpload}>Reflect</button>
      {imageUrl && <img src={imageUrl} alt="Reflected" />}
    </div>
  );
}
```

#### Using a Next.js API Route as Proxy

Create `pages/api/reflect.ts`:

```ts
// Disable Next.js body parsing to forward form-data
export const config = { api: { bodyParser: false } };

import type { NextApiRequest, NextApiResponse } from "next";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const targetUrl = `http://localhost:8000${req.url}`;
  const apiRes = await fetch(targetUrl, {
    method: req.method,
    headers: req.headers as any,
    body: req.body,
  });
  const data = await apiRes.arrayBuffer();
  res.status(apiRes.status);
  res.setHeader("Content-Type", apiRes.headers.get("content-type") || "");
  res.send(Buffer.from(data));
}
```

## Development

- Core image-processing logic lives in `purecv.py`.
- API definitions are in `main.py`.
- Tests can be added under a `tests/` directory using `pytest` and FastAPI's `TestClient`.

Build and run:

```bash
docker build -t purecv-service .
docker run -d -p 8000:8000 purecv-service
```

## Configuration

Environment variables (e.g., `PORT`, logging levels) can be added as needed.
# astra_python
