'use client';

import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { TierSelector } from '@/components/TierSelector';
import { Tier } from '@/lib/types';
import { Plus, Trash2, Upload, CheckCircle, Sparkles } from 'lucide-react';

export default function DeployPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    name: '',
    tier: 'FAST' as Tier,
    gpu_count: 1,
    vram_per_gpu_gb: 40,
    cpu_cores: 4,
    ram_gb: 16,
    image: '',
    command: '',
    env: [{ key: '', value: '' }],
  });

  const [containerMode, setContainerMode] = useState<'manual' | 'upload' | 'demo'>('manual');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const gpuPresets = [1, 8, 16, 32, 64, 128, 256, 512];

  const handleEnvChange = (index: number, field: 'key' | 'value', value: string) => {
    const newEnv = [...formData.env];
    newEnv[index][field] = value;
    setFormData({ ...formData, env: newEnv });
  };

  const addEnvVar = () => {
    setFormData({ ...formData, env: [...formData.env, { key: '', value: '' }] });
  };

  const removeEnvVar = (index: number) => {
    const newEnv = formData.env.filter((_, i) => i !== index);
    setFormData({ ...formData, env: newEnv.length > 0 ? newEnv : [{ key: '', value: '' }] });
  };

  const handleFileUpload = (file: File) => {
    if (file.type === 'application/x-tar' || file.name.endsWith('.tar') || 
        file.name.endsWith('.tar.gz') || file.name.endsWith('.dockerfile') ||
        file.type === 'text/plain') {
      setUploadedFile(file);
      setContainerMode('upload');
      setFormData({ ...formData, image: `uploaded:${file.name}` });
    } else {
      alert('Please upload a valid Docker image (.tar, .tar.gz) or Dockerfile');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const useDemoContainer = () => {
    setContainerMode('demo');
    setUploadedFile(null);
    setFormData({
      ...formData,
      image: 'demo/pytorch-gpu:latest',
      command: '["python", "demo_train.py"]',
      env: [
        { key: 'PYTORCH_VERSION', value: '2.0' },
        { key: 'CUDA_VERSION', value: '11.8' },
      ],
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Parse command (comma-separated or JSON array)
      let commandArray: string[] | undefined;
      if (formData.command) {
        try {
          commandArray = JSON.parse(formData.command);
        } catch {
          commandArray = formData.command.split(',').map((s) => s.trim()).filter(Boolean);
        }
      }

      // Parse environment variables
      const envObj: Record<string, string> = {};
      formData.env.forEach(({ key, value }) => {
        if (key && value) {
          envObj[key] = value;
        }
      });

      await api.createApp({
        name: formData.name,
        tier: formData.tier,
        gpu_count: formData.gpu_count,
        vram_per_gpu_gb: formData.vram_per_gpu_gb,
        cpu_cores: formData.cpu_cores,
        ram_gb: formData.ram_gb,
        image: formData.image,
        command: commandArray,
        env: Object.keys(envObj).length > 0 ? envObj : undefined,
      });

      router.push('/apps');
    } catch (err) {
      console.error('Deployment error:', err);
      setError(err instanceof Error ? err.message : JSON.stringify(err) || 'Failed to deploy application');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-8 py-12">
      <div className="mb-8">
        <h1 className="text-4xl font-serif font-bold text-gray-900 mb-2">
          Deploy Application
        </h1>
        <p className="text-gray-600">
          Configure and deploy your GPU compute application
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* App Details */}
        <div className="rounded-xl p-6 bg-gray-50">
          <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
            Application Details
          </h2>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Application Name
            </label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
              placeholder="my-training-job"
            />
          </div>
        </div>

        {/* Tier Selection */}
        <div className="rounded-xl p-6 bg-gray-50">
          <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
            Compute Tier
          </h2>
          <TierSelector
            selected={formData.tier}
            onChange={(tier) => setFormData({ ...formData, tier })}
          />
        </div>

        {/* Resource Configuration */}
        <div className="rounded-xl p-6 bg-gray-50">
          <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
            Resource Configuration
          </h2>
          <div className="space-y-4">
            {/* GPU Count */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                GPU Count
              </label>
              <div className="flex flex-wrap gap-2 mb-2">
                {gpuPresets.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setFormData({ ...formData, gpu_count: preset })}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      formData.gpu_count === preset
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {preset}
                  </button>
                ))}
              </div>
              <input
                type="number"
                min="1"
                value={formData.gpu_count}
                onChange={(e) =>
                  setFormData({ ...formData, gpu_count: parseInt(e.target.value) })
                }
                className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
              />
            </div>

            {/* VRAM per GPU */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                VRAM per GPU (GB)
              </label>
              <input
                type="number"
                min="1"
                step="0.1"
                value={formData.vram_per_gpu_gb}
                onChange={(e) =>
                  setFormData({ ...formData, vram_per_gpu_gb: parseFloat(e.target.value) })
                }
                className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
              />
            </div>

            {/* CPU Cores */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                CPU Cores
              </label>
              <input
                type="number"
                min="1"
                value={formData.cpu_cores}
                onChange={(e) =>
                  setFormData({ ...formData, cpu_cores: parseInt(e.target.value) })
                }
                className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
              />
            </div>

            {/* RAM */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                RAM (GB)
              </label>
              <input
                type="number"
                min="1"
                step="0.1"
                value={formData.ram_gb}
                onChange={(e) =>
                  setFormData({ ...formData, ram_gb: parseFloat(e.target.value) })
                }
                className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
              />
            </div>
          </div>
        </div>

        {/* Container Configuration */}
        <div className="rounded-xl p-6 bg-gray-50">
          <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
            Container Setup
          </h2>

          {/* Mode Selection */}
          <div className="flex gap-3 mb-6">
            <button
              type="button"
              onClick={() => setContainerMode('manual')}
              className={`flex-1 px-4 py-3 rounded-lg transition-all ${
                containerMode === 'manual'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="font-medium">Docker Image</div>
              <div className="text-xs mt-1 opacity-75">Use existing image</div>
            </button>
            <button
              type="button"
              onClick={() => setContainerMode('upload')}
              className={`flex-1 px-4 py-3 rounded-lg transition-all ${
                containerMode === 'upload'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="font-medium">Upload Container</div>
              <div className="text-xs mt-1 opacity-75">Drag & drop file</div>
            </button>
            <button
              type="button"
              onClick={useDemoContainer}
              className={`flex-1 px-4 py-3 rounded-lg transition-all ${
                containerMode === 'demo'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="flex items-center justify-center gap-1 font-medium">
                <Sparkles className="w-4 h-4" />
                Demo Container
              </div>
              <div className="text-xs mt-1 opacity-75">Quick start</div>
            </button>
          </div>

          <div className="space-y-4">
            {/* Manual Docker Image Input */}
            {containerMode === 'manual' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Docker Image
                </label>
                <input
                  type="text"
                  required
                  value={formData.image}
                  onChange={(e) => setFormData({ ...formData, image: e.target.value })}
                  className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  placeholder="nvidia/cuda:11.8.0-base-ubuntu22.04"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Public Docker Hub image or private registry URL
                </p>
              </div>
            )}

            {/* Drag and Drop Upload */}
            {containerMode === 'upload' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Container Image
                </label>
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-indigo-600 bg-indigo-50'
                      : uploadedFile
                      ? 'border-green-400 bg-green-50'
                      : 'border-gray-300 bg-white hover:border-gray-400'
                  }`}
                >
                  {uploadedFile ? (
                    <div className="flex flex-col items-center gap-2">
                      <CheckCircle className="w-12 h-12 text-green-600" />
                      <div className="font-medium text-gray-900">{uploadedFile.name}</div>
                      <div className="text-sm text-gray-600">
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setUploadedFile(null);
                          setFormData({ ...formData, image: '' });
                        }}
                        className="mt-2 text-sm text-red-600 hover:text-red-700 font-medium"
                      >
                        Remove file
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="w-12 h-12 text-gray-400" />
                      <div className="font-medium text-gray-900">
                        Drop your container image here
                      </div>
                      <div className="text-sm text-gray-600">
                        or click to browse
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Supports .tar, .tar.gz, or Dockerfile
                      </div>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tar,.tar.gz,.dockerfile,text/plain"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            )}

            {/* Demo Container Info */}
            {containerMode === 'demo' && (
              <div className="rounded-xl p-4 bg-indigo-50">
                <div className="flex items-start gap-3">
                  <Sparkles className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="font-medium text-indigo-900 mb-1">
                      Demo Container Loaded
                    </div>
                    <p className="text-sm text-indigo-700 mb-2">
                      PyTorch 2.0 with CUDA 11.8 - Perfect for testing and demonstration
                    </p>
                    <div className="text-xs text-indigo-600 font-mono bg-white/50 rounded px-2 py-1">
                      demo/pytorch-gpu:latest
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Command */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Command (optional)
              </label>
              <input
                type="text"
                value={formData.command}
                onChange={(e) => setFormData({ ...formData, command: e.target.value })}
                className="w-full bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                placeholder='["python", "train.py"] or python, train.py'
              />
              <p className="mt-1 text-xs text-gray-500">
                JSON array or comma-separated values
              </p>
            </div>

            {/* Environment Variables */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Environment Variables
              </label>
              <div className="space-y-2">
                {formData.env.map((envVar, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="text"
                      value={envVar.key}
                      onChange={(e) => handleEnvChange(index, 'key', e.target.value)}
                      className="flex-1 bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                      placeholder="KEY"
                    />
                    <input
                      type="text"
                      value={envVar.value}
                      onChange={(e) => handleEnvChange(index, 'value', e.target.value)}
                      className="flex-1 bg-white rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                      placeholder="value"
                    />
                    <button
                      type="button"
                      onClick={() => removeEnvVar(index)}
                      className="p-2 text-gray-600 hover:text-red-600 transition-colors"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={addEnvVar}
                className="mt-2 inline-flex items-center gap-1 text-sm text-indigo-600 hover:text-indigo-700 font-medium"
              >
                <Plus className="w-4 h-4" />
                Add variable
              </button>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="rounded-xl p-4 bg-red-50 text-red-800">
            {error}
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end gap-4">
          <button
            type="button"
            onClick={() => router.back()}
            className="px-6 py-3 rounded-lg font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-8 py-3 rounded-lg font-medium bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Deploying...' : 'Deploy Application'}
          </button>
        </div>
      </form>
    </div>
  );
}

